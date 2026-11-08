import pandas as pd, numpy as np, random, collections
S0=pd.read_csv('samples.csv')
def seq(f): return ''.join(l.strip() for l in open(f) if not l.startswith('>')).upper().replace('T','U')
G={'KUNV':seq('../refs/WNV_KUN_NSW2011.fa'),'NY99':seq('../refs/WNV_NY99.fa'),'JEV':seq('../refs/JEV_FU.fa')}
def load(g):
    S=S0[(S0.group==g)&(S0.role=='infected')].sort_values('replicate'); R=[]
    for r in S.itertuples():
        d=pd.read_csv('aligned_tables/'+r.file,sep='\t',header=None,names=['name','flag','ref','pos','cigar','seq'])
        d=d[(d.flag==0)&d.cigar.astype(str).str.fullmatch(r'\d+M')].copy()
        d['count']=d.name.astype(str).str.rsplit('-',n=1).str[1].astype(int)
        d['end']=d.pos+d.seq.str.len()-1
        R.append((d.groupby(['pos','end'])['count'].sum()/float(r.total_reads)*1e6).to_dict())
    return R
reads={v:load('HEK_'+v) for v in G}
F=pd.read_csv('cmpstruct/fragments_by_element.csv'); F=F[(F.near_CPM>=20)&(F.length<=100)]
print('fragments',len(F),F.groupby('virus').size().to_dict())
def best(v,a,b):
    R=reads[v]; keys=set().union(*[r.keys() for r in R])
    w={(p,e):np.mean([r.get((p,e),0) for r in R]) for (p,e) in keys if abs(p-a)<=2 and abs(e-b)<=2 and e-p+1>=14}
    return max(w,key=w.get)
STALL={'KUNV':[10494,10654,10745,10822],'NY99':[10508,10668,10759,10836],'JEV':[10446,10605,10692,10773]}
EXCL=True
def cuts(rows,mode):
    out={v:set() for v in G}
    for r in rows.itertuples():
        a,b=int(r.start),int(r.end)
        if mode=='top': a,b=best(r.virus,a,b)
        L=len(G[r.virus])
        if not (EXCL and any(abs(a-st)<=3 for st in STALL[r.virus])):
            out[r.virus].add(a-1)            # cut after base index a-1 (1-based): between a-1 and a
        else: print('excluded 5prime cut',r.virus,a,b)
        if b<L: out[r.virus].add(b)      # between b and b+1
    return out
def feats(v,x):  # x = 1-based position of base before the cut
    s=G[v]; p=s[x-1]; q=s[x]; pp=s[x-2] if x>=2 else 'N'
    return dict(UAAU=(p+q) in ('UA','AU'), Ueither=(p=='U' or q=='U'), Ubefore=(p=='U'),
        RNaseL=(pp=='U' and p in 'UA'), YR=(p in 'UC' and q in 'AG'), UR=(p=='U' and q in 'AG'),
        RU=(p in 'AG' and q=='U'), Abefore=(p=='A'), AU_only=(p+q)=='AU', UA_only=(p+q)=='UA')
def summarise(cs):
    rows=[(v,x) for v,xs in cs.items() for x in xs if 1<x<len(G[v])-1]
    df=pd.DataFrame([feats(v,x) for v,x in rows]); return df,rows
random.seed(1)
def rand_stats(cs,n=2000):
    res=collections.defaultdict(list)
    for _ in range(n):
        rows=[]
        for v,xs in cs.items():
            L=len(G[v]); pool=list(range(L-620,L-1))
            rows+= [(v,x) for x in random.sample(pool,len([x for x in xs if 1<x<L-1]))]
        d=pd.DataFrame([feats(v,x) for v,x in rows]).mean()
        for k,val in d.items(): res[k].append(val)
    return res
for mode in ('peak',):
    cs=cuts(F,mode); df,rows=summarise(cs); rs=rand_stats(cs)
    print(f'\n=== ends: {mode}  | distinct cuts {len(df)}')
    print(f"{'feature':8s} {'observed':>8s} {'random':>7s} {'p(>=obs)':>9s}")
    for k in df.columns:
        o=df[k].mean(); rnd=np.mean(rs[k]); p=(np.sum(np.array(rs[k])>=o-1e-12)+1)/(len(rs[k])+1)
        print(f'{k:8s} {o*100:7.1f}% {rnd*100:6.1f}% {p:9.3f}')
    dn=collections.Counter(G[v][x-1]+G[v][x] for v,x in rows)
    print('dinucleotide at cut (before|after):',dict(sorted(dn.items(),key=lambda t:-t[1])))
    pb=collections.Counter(G[v][x-1] for v,x in rows); pa=collections.Counter(G[v][x] for v,x in rows)
    print('base before cut',dict(pb),' base after',dict(pa))
    if mode=='peak': base=set((v,x) for v,x in rows)
    else: print('cuts changed by using top pair:',len(set((v,x) for v,x in rows)-base),'of',len(rows))

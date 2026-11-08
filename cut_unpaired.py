# Unpaired-nucleotide test for cut sites: all distinct cuts vs 2,000 random sets (same design as cutcheck.py)
import sys, random, collections, re
import numpy as np, pandas as pd
src=open('robust/cutcheck.py').read().split("for mode in ('peak','top'):")[0]
exec(src)                                    # G, reads, F, cuts()
W=620; D='../unp/'
def parse_pf(v):                              # global fold: MFE structure + base-pair probabilities from dp.ps
    st=open(D+v+'.fold.txt').read().split('\n')[2].split(' ')[0]
    P=np.zeros(W)
    for l in open(D+v+'_dp.ps'):
        m=re.match(r'^(\d+) (\d+) ([\d.]+) ubox',l)
        if m:
            i,j,p=int(m.group(1))-1,int(m.group(2))-1,float(m.group(3))**2; P[i]+=p; P[j]+=p
    return np.array([c=='.' for c in st]),1-P
def parse_l(v):
    return np.array([float(l.split()[1]) for l in open(D+v+'_lunp') if re.match(r'^\d+\s',l)])
MF={};PG={};PL={}
for v in G: MF[v],PG[v]=parse_pf(v); PL[v]=parse_l(v)
def score(v,x):                               # cut between 1-based genome positions x and x+1
    L=len(G[v]); i=x-(L-W)-1
    if i<0 or i+1>=W: return None
    return dict(mfe_both=float(MF[v][i] and MF[v][i+1]), pu_g=PG[v][i]*PG[v][i+1], pu_l=PL[v][i]*PL[v][i+1])
cs=cuts(F,'peak')
rows=[(v,x) for v,xs in cs.items() for x in xs if 1<x<len(G[v])-1]
od=pd.DataFrame([s for s in (score(v,x) for v,x in rows) if s]); print('distinct cuts',len(rows),'scored',len(od))
random.seed(7); R=collections.defaultdict(list)
for _ in range(2000):
    sc=[]
    for v,xs in cs.items():
        L=len(G[v]); k=len([x for x in xs if 1<x<L-1]); pool=list(range(L-620+1,L-1))
        sc+=[score(v,x) for x in random.sample(pool,k)]
    d=pd.DataFrame([s for s in sc if s]).mean()
    for kk,val in d.items(): R[kk].append(val)
lab={'mfe_both':'both nt unpaired in MFE structure','pu_g':'P(both unpaired), global fold','pu_l':'P(both unpaired), local fold'}
print(f"{'measure':36s}{'observed':>9s}{'random':>8s}{'p(>=obs)':>10s}")
for kk in lab:
    o=od[kk].mean(); r=np.mean(R[kk]); p=(np.sum(np.array(R[kk])>=o-1e-12)+1)/(len(R[kk])+1)
    print(f'{lab[kk]:36s}{o:9.3f}{r:8.3f}{p:10.4f}')
for v in G:
    ss=[s for s in (score(v,x) for x in cs[v] if 1<x<len(G[v])-1) if s]
    print(v,len(ss),'cuts  MFE both unpaired',round(np.mean([s['mfe_both'] for s in ss]),3),' local P',round(np.mean([s['pu_l'] for s in ss]),3))

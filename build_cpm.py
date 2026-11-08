import re, sys, numpy as np, pandas as pd
cols=["name","flag","ref","pos","cigar","seq"]
op=re.compile(r'(\d+)([MIDNSHP=X])')
def span(c):
    return sum(int(n) for n,o in op.findall(c) if o in 'MDN=X')
def counts(path, L):
    d=pd.read_csv(path,sep='\t',header=None,names=cols)
    d=d[d.flag==0].copy()
    d['c']=d.name.astype(str).str.rsplit('-',n=1).str[1].astype(int)
    d['end']=d.pos+d.cigar.map(span)-1
    f5=np.bincount(d.pos,weights=d.c,minlength=L+2)[1:L+1]
    f3=np.bincount(d.end.clip(upper=L),weights=d.c,minlength=L+2)[1:L+1]
    diff=np.zeros(L+3)
    np.add.at(diff,d.pos.values,d.c.values); np.add.at(diff,(d.end.values+1).clip(max=L+1),-d.c.values)
    cov=np.cumsum(diff)[1:L+1]
    return f5,f3,cov
def build(group, L, samples, aligned='aligned_tables', out='results'):
    S=samples[samples.group==group]
    res={'5prime':{}, '3prime':{}, 'coverage':{}}
    for role in ('infected','mock'):
        for r in S[S.role==role].sort_values('replicate').itertuples():
            import os
            p=f'{aligned}/{r.file}'
            if not os.path.exists(p): f5=f3=cov=np.zeros(L)
            else: f5,f3,cov=counts(p,L)
            for k,v in zip(('5prime','3prime','coverage'),(f5,f3,cov)):
                res[k][f'{role}_rep{r.replicate}_CPM']=v/float(r.total_reads)*1e6
    for k,dct in res.items():
        df=pd.DataFrame(dct); df.insert(0,'position',np.arange(1,L+1))
        for role in ('infected','mock'):
            c=[x for x in df.columns if x.startswith(role+'_rep')]
            df[f'{role}_mean_CPM']=df[c].mean(axis=1); df[f'{role}_SD']=df[c].std(axis=1,ddof=1)
        order=['position']+[f'infected_rep{i}_CPM' for i in (1,2,3)]+['infected_mean_CPM','infected_SD']+[f'mock_rep{i}_CPM' for i in (1,2,3)]+['mock_mean_CPM','mock_SD']
        yield k, df[order]
if __name__=='__main__':
    group,L,outdir=sys.argv[1],int(sys.argv[2]),sys.argv[3]
    samples=pd.read_csv('samples.csv')
    import os; os.makedirs(outdir,exist_ok=True)
    for k,df in build(group,L,samples):
        df.to_csv(f'{outdir}/{group}_{k}_CPM.csv',index=False)

import pandas as pd, numpy as np, matplotlib, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','TeX Gyre Termes'],'font.size':13,
 'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':14,'axes.labelsize':13.5,'xtick.labelsize':12,'ytick.labelsize':12,
 'legend.fontsize':12,'svg.fonttype':'path','mathtext.fontset':'stix'})
OUT='final2/'; os.makedirs(OUT,exist_ok=True)
S=pd.read_csv('samples.csv')
BLUE='#1f6fd0'; OR='#e0531a'; GR='#17a673'; SH='#fcf0dc'
KEL=[('SL-1',10407,10473),('SL-2 (xrRNA1)',10494,10542),('SL-3',10580,10641),('SL-4 (xrRNA2)',10654,10696),('DB-1',10745,10813),('DB-2',10822,10887),('sHP',10921,10936),("3′SL",10937,11015)]
JEL=[('SL-1',10412,10440),('xrRNA1',10446,10496),('SL-3',10528,10600),('xrRNA2',10605,10651),('DB-1',10692,10762),('DB-2',10773,10840),("3′SL",10881,10964)]
NEL=[('SL-2 (xrRNA1)',10508,10556),('SL-3',10594,10655),('SL-4 (xrRNA2)',10668,10710),('DB-1',10759,10827),('DB-2',10836,10900),("3′SL",10951,11029)]
CFG={'HEK_KUNV':('HEK293 KUNV',10399,11015,KEL),'A549_KUNV':('A549 KUNV',10399,11015,KEL),'HEK_JEV':('HEK293 JEV',10395,10964,JEL),'HEK_NY99':('HEK293 NY99',10399,11029,NEL)}
def save(fig,name):
    fig.savefig(f'{OUT}{name}.png',dpi=300,bbox_inches='tight'); fig.savefig(f'{OUT}{name}.svg',bbox_inches='tight'); plt.close(fig)
# ---------- Fig 3.2 ----------
def fig32(g):
    title,r0,r1,els=CFG[g]
    cov=pd.read_csv(f'results/{g}_coverage_CPM.csv').set_index('position'); pr=pd.read_csv(f'peaks_new/{g}_fragment_pairs.csv')
    top=pr.sort_values(['exact_reads_CPM','pair_score_min_CPM'],ascending=False).head(12).iloc[::-1]
    fig,ax=plt.subplots(2,1,figsize=(9.5,8.2),sharex=True,gridspec_kw={'height_ratios':[1,1.6],'hspace':0.07})
    for a in ax:
        for n,s,e in els: a.axvspan(s,e,color=SH,lw=0,zorder=0)
        a.grid(axis='y',color='#e5e5e5'); a.set_axisbelow(True)
    for k,(n,s,e) in enumerate(els): ax[0].text((s+e)/2,1.02+(0.075 if k%2 else 0),n,transform=ax[0].get_xaxis_transform(),ha='center',va='bottom',fontsize=10.5)
    x=np.arange(r0,r1+1); m=cov.loc[r0:r1,'infected_mean_CPM']; sd=cov.loc[r0:r1,'infected_SD']
    ax[0].fill_between(x,m-sd,m+sd,color='#a9c8ee',alpha=.55,lw=0,label='± 1 SD'); ax[0].plot(x,m,color=BLUE,lw=2,label='infected (mean, n = 3)')
    ax[0].plot(x,cov.loc[r0:r1,'mock_mean_CPM'],color='#888',lw=1.5,label='mock'); ax[0].set_ylabel('sense coverage (CPM)'); ax[0].set_ylim(0,(m+sd).max()*1.35)
    ax[0].legend(loc='upper left',frameon=False,ncol=3,bbox_to_anchor=(0,1.0),columnspacing=1.2,handlelength=1.6)
    a=ax[1]; n=len(top); a.set_ylim(-.8,n-.2); a.set_yticks([])
    for i,(_,r) in enumerate(top.iterrows()):
        s,e=int(r.start_5prime_pos),int(r.end_3prime_pos); L=e-s+1; sup=bool(r.supported)
        a.barh(i,L,left=s,height=.64,color='#8b7b62' if sup else 'none',edgecolor='#8b7b62',hatch=None if sup else '///',alpha=.95 if sup else .6,lw=1.2)
        a.barh(i,1.8,left=s,height=.64,color=OR); a.barh(i,1.8,left=e-.8,height=.64,color=GR)
        v=r.exact_reads_CPM; lab=f'{L} nt · {v:.0f} CPM' if v>=10 else f'{L} nt · {v:.1f} CPM'
        a.text(e+4,i,lab,va='center',fontsize=11.5,color='#222' if sup else '#777')
    a.set_xlabel('genome position (nt)'); a.set_xlim(r0-8,r1+(100 if r1-r0>600 else 85))
    a.set_ylabel('top 12 candidate fragments, ranked by CPM')
    ax[0].set_title(f'{title}',loc='left',pad=46,fontsize=16,fontweight='bold')
    note='Orange: fragment 5′ end. Green: 3′ end. Number after each bar: length and CPM of reads\nthat start and end exactly at the two ends. Solid bar: at least 5 CPM; hatched bar: below 5 CPM.'
    a.text(0,-0.185,note,transform=a.transAxes,fontsize=10.5,ha='left',va='top'); save(fig,f'Fig3.2_{g}_coverage_fragments')
for g in CFG: fig32(g)
# ---------- Fig 3.5 ----------
fig,ax=plt.subplots(4,1,figsize=(9.5,11.5),sharex=True,gridspec_kw={'hspace':0.5})
for a,(g,(title,u0,u1,_)) in zip(ax,CFG.items()):
    cov=pd.read_csv(f'results/{g}_coverage_CPM.csv').set_index('position')
    m=cov.infected_mean_CPM; sd=cov.infected_SD; x=m.index
    a.axvspan(u0,u1,color=SH,lw=0); a.fill_between(x,m-sd,m+sd,color='#a9c8ee',alpha=.55,lw=0); a.plot(x,m,color=BLUE,lw=1.4)
    a.set_xlim(0,11100); a.set_ylim(0,m.max()*1.22); a.set_ylabel('CPM'); a.set_title(title,loc='left',fontsize=14,fontweight='bold',pad=16)
    a.text((u0+u1)/2,1.01,'3′ UTR',transform=a.get_xaxis_transform(),ha='center',va='bottom',fontsize=11.5); a.grid(axis='y',color='#e5e5e5'); a.set_axisbelow(True)
    a.text(0.005,0.9,f'mean coverage: 3′ UTR {m.loc[u0:u1].mean():.0f} CPM; rest of genome {m.loc[:u0-1].mean():.0f} CPM',transform=a.transAxes,ha='left',fontsize=11,color='#444')
ax[-1].set_xlabel('genome position (nt)')
ax[-1].text(0,-0.42,'Sense reads only. Mean of 3 infected replicates ± 1 SD, in counts per million library reads (CPM). Each panel has its own y-scale.',transform=ax[-1].transAxes,fontsize=10.5,ha='left',va='top')
save(fig,'Fig3.5_whole_genome_sense_coverage')
# ---------- Fig 3.1: one figure per condition, pair of panels ----------
cols=["name","flag","ref","pos","cigar","seq"]
def lens(g,u0):
    s=S[(S.group==g)&(S.role=='infected')].sort_values('replicate'); A=[];U=[]
    for r in s.itertuples():
        d=pd.read_csv(f'aligned_tables/{r.file}',sep='\t',header=None,names=cols); d=d[d.flag==0].copy()
        d['c']=d.name.str.rsplit('-',n=1).str[1].astype(int); d['L']=d.seq.str.len()
        A.append(np.bincount(d.L,weights=d.c,minlength=300)/r.total_reads*1e6); u=d[d.pos>=u0]; U.append(np.bincount(u.L,weights=u.c,minlength=300)/r.total_reads*1e6)
    return np.array(A),np.array(U)
for g in CFG:
    title,u0,u1,_=CFG[g]; A,U=lens(g,u0)
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.2),gridspec_kw={'wspace':.28})
    for a,D,lab in zip(ax,(A,U),('All aligned reads',f'3′ UTR reads ({u0}–{u1})')):
        xs=np.arange(15,61); m=D.mean(0)[15:61]; s=D.std(0,ddof=1)[15:61]
        a.bar(xs,m,color=BLUE,width=.8,zorder=2); a.errorbar(xs,m,yerr=s,fmt='none',ecolor='#444',lw=.9,capsize=1.5,zorder=3)
        a.set_xlim(14.2,60.8); a.set_ylim(0,(m+s).max()*1.13); a.grid(axis='y',color='#e5e5e5'); a.set_axisbelow(True)
        a.set_title(lab,loc='left',fontsize=13.5); a.set_xlabel('read length (nt)'); a.set_ylabel('reads per million (CPM)')
        a.set_xticks([15,20,30,40,50,60])
        tt=list(np.argsort(m)[::-1][:2]); close=abs(tt[0]-tt[1])==1
        for t in tt:
            dx=(-0.45 if t==min(tt) else 0.45) if close else 0
            a.text(xs[t]+dx,m[t]+s[t]+m.max()*.012,str(xs[t]),ha='center',fontsize=11)
    fig.suptitle(title+', sense reads',x=.01,ha='left',y=1.04,fontsize=15,fontweight='bold')
    fig.text(.01,-.06,'Mean ± SD of 3 infected replicates; reads of 15 nt and longer.',fontsize=10.5)
    save(fig,f'Fig3.1_{g}_read_length')

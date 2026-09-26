import json, csv, numpy as np, matplotlib
import os
REPO=os.environ.get('ATML_REPO','.')
OUT='report/figures/task3'; os.makedirs(OUT,exist_ok=True)
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.grid':True,'grid.alpha':0.3,'pdf.fonttype':42})
T3=REPO+'/task3/results'; T2=REPO+'/task2/results'
d=json.load(open(f'{T3}/final/final_results.json')); R=d['results']
cls=['dog','elephant','giraffe','guitar','horse','house','person']
def hist(p): return list(csv.DictReader(open(p)))
H={'erm':hist(f'{T2}/training/source_only/history.csv')}
for m in ['dan_dg_0p1','dan_dg_1','dan_dg_10','sam']: H[m]=hist(f'{T3}/training/prescribed/{m}/history.csv')
H['dan_dg_floor_1']=hist(f'{T3}/training/supplementary/dan_dg_floor_1/history.csv')
col=lambda h,k:[float(x[k]) for x in h]
ep=lambda h:[int(x['epoch']) for x in h]
sel={m:R[m]['identity']['selected_epoch'] for m in R}
C={'erm':'0.35','dan_dg_1':'tab:blue','sam':'tab:orange','dan_dg_0p1':'tab:cyan','dan_dg_10':'tab:red','task2':'tab:green'}
L={'erm':'ERM','dan_dg_1':r'DAN-DG $\lambda{=}1$','sam':r'SAM $\rho{=}0.05$','dan_dg_0p1':r'DAN-DG $\lambda{=}0.1$','dan_dg_10':r'DAN-DG $\lambda{=}10$'}

# ---------- Figure A: training curves ----------
fig,ax=plt.subplots(1,3,figsize=(7.2,2.15))
for m in ['erm','dan_dg_1','sam']:
    ax[0].plot(ep(H[m]),col(H[m],'classification_loss'),'-o',ms=2.5,lw=1.2,color=C[m],label=L[m])
    ax[0].axvline(sel[m],color=C[m],ls=':',lw=0.9)
ax[0].plot(ep(H['sam']),col(H['sam'],'sam_perturbed_classification_loss'),'--',lw=1,color=C['sam'],label=r'SAM at $\theta{+}\epsilon$')
ax[0].set_title('Source classification loss'); ax[0].set_xlabel('source epoch'); ax[0].legend(frameon=False,loc='upper right')
for m in ['dan_dg_0p1','dan_dg_1','dan_dg_10']:
    ax[1].plot(ep(H[m]),col(H[m],'mmd_loss'),'-o',ms=2.5,lw=1.2,color=C[m],label=L[m])
    ax[1].axvline(sel[m],color=C[m],ls=':',lw=0.9)
ax[1].set_title(r'Mean pairwise MMD$^2$ (unweighted)'); ax[1].set_xlabel('source epoch'); ax[1].legend(frameon=False,loc='upper right',fontsize=6.5)
for m in ['erm','dan_dg_0p1','dan_dg_1','dan_dg_10','sam']:
    ax[2].plot(ep(H[m]),col(H[m],'gradient_norm'),'-o',ms=2.5,lw=1.2,color=C[m],label=L[m])
ax[2].axhline(20,color='k',ls='--',lw=0.8); ax[2].text(12.2,24,'clip = 20',fontsize=6.5,ha='right')
ax[2].set_yscale('log'); ax[2].set_ylim(2,3e6); ax[2].set_title('Mean pre-clipping gradient norm'); ax[2].set_xlabel('source epoch')
ax[2].legend(frameon=False,fontsize=5.8,loc='upper left',ncol=2,columnspacing=0.8,handlelength=1.5)
from matplotlib.ticker import MaxNLocator
for a_ in ax: a_.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout(pad=0.3,w_pad=0.8)
for e in ['pdf','png']: fig.savefig(f'{OUT}/task3_training_curves.{e}',dpi=300,bbox_inches='tight')

# ---------- Figure B: per-class change + controlled study ----------
def acc(cm): cm=np.array(cm); return 100*cm.diagonal()/cm.sum(1)
erm=acc(R['erm']['target']['confusion_matrix'])
t2=json.load(open(f'{T2}/final/confusion_matrices.json'))
bars=[('dan_dg_1',acc(R['dan_dg_1']['target']['confusion_matrix'])-erm,C['dan_dg_1'],L['dan_dg_1'],None),
      ('sam',acc(R['sam']['target']['confusion_matrix'])-erm,C['sam'],L['sam'],None),
      ('t2',acc(t2['dan_1'])-erm,C['task2'],r'Task 2 DAN $\lambda{=}1$ (target-aware)','//')]
fig=plt.figure(figsize=(7.2,4.0))
gs=fig.add_gridspec(2,4,height_ratios=[1.15,1],hspace=0.55,wspace=0.42)
a=fig.add_subplot(gs[0,:]); x=np.arange(7); w=0.26
for i,(k,v,c,l,h) in enumerate(bars):
    a.bar(x+(i-1)*w,v,w,color=c,label=l,hatch=h,edgecolor='white' if h is None else 'k',lw=0.3)
a.axhline(0,color='k',lw=0.6); a.set_xticks(x); a.set_xticklabels([f'{c}\n(ERM {e:.0f}%)' for c,e in zip(cls,erm)])
a.set_ylabel('Sketch class-accuracy\nchange vs ERM (pp)'); a.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.16)); a.grid(axis='x',alpha=0)
lam=[0.1,1,10]; ms=['dan_dg_0p1','dan_dg_1','dan_dg_10']
def sv(m,k): return 100*R[m]['source_validation'][k]
panels=[('Source macro-F1 (%)',lambda m:sv(m,'mean_source_macro_f1'),lambda m:sv(m,'worst_source_macro_f1'),'linear'),
        ('Source-domain separability (%)',lambda m:100*R[m]['source_diagnostics']['source_domain_separability']['accuracy'],None,'linear'),
        (r'Sharpness proxy $\Delta_{\rm sharp}$',lambda m:R[m]['source_diagnostics']['common_sharpness_proxy']['delta_sharp'],None,'log'),
        ('Sketch (%)',lambda m:100*R[m]['target']['accuracy'],lambda m:100*R[m]['target']['macro_f1'],'linear')]
for j,(t,f,f2,sc) in enumerate(panels):
    b=fig.add_subplot(gs[1,j])
    b.plot(lam,[f(m) for m in ms],'-o',ms=3.5,color=C['dan_dg_1'],label='mean' if j==0 else ('acc.' if j==3 else None))
    b.axhline(f('erm'),color='0.4',ls='--',lw=0.9)
    if f2:
        b.plot(lam,[f2(m) for m in ms],'-s',ms=3,color=C['dan_dg_1'],alpha=0.45,label='worst' if j==0 else 'macro-F1')
        b.axhline(f2('erm'),color='0.4',ls=':',lw=0.9)
        b.legend(frameon=False,fontsize=6.5,loc='lower left')
    if j==1: b.axhline(100/3,color='k',ls=':',lw=0.7); b.text(0.11,36,'chance',fontsize=6)
    b.set_xscale('log'); b.set_yscale(sc); b.set_xticks(lam); b.set_xticklabels(['0.1','1','10']); b.set_xlabel(r'$\lambda_{\rm DG}$'); b.set_title(t,fontsize=7.5)
for e in ['pdf','png']: fig.savefig(f'{OUT}/task3_class_and_strength.{e}',dpi=300,bbox_inches='tight')

# ---------- Appendix figure: bandwidth collapse ----------
fig,ax=plt.subplots(1,2,figsize=(7.2,2.1))
pairs=[('photo__art_painting','P–A'),('photo__cartoon','P–C'),('art_painting__cartoon','A–C')]
for m in ['dan_dg_0p1','dan_dg_1','dan_dg_10']:
    med=np.mean([[float(x[f'mmd_median_{p}']) for p,_ in pairs] for x in H[m]],axis=1)
    med=np.where(med<=0,np.nan,med)
    ax[0].plot(ep(H[m]),med,'-o',ms=2.5,color=C[m],label=L[m])
    ax[1].plot(ep(H[m]),col(H[m],'gradient_clipped_fraction'),'-o',ms=2.5,color=C[m],label=L[m])
fl=H['dan_dg_floor_1']
ax[0].plot(ep(fl),np.mean([[float(x[f'mmd_effective_median_{p}']) for p,_ in pairs] for x in fl],axis=1),'--',color=C['dan_dg_1'],lw=1,label=r'floor, $\lambda{=}1$ (effective)')
ax[1].plot(ep(fl),col(fl,'gradient_clipped_fraction'),'--',color=C['dan_dg_1'],lw=1,label=r'floor variant $\lambda{=}1$')
ax[0].set_yscale('log'); ax[0].set_title('Epoch-mean batch median sq. distance (MMD bandwidth scale)'); ax[0].set_xlabel('source epoch')
ax[1].set_title('Fraction of updates clipped'); ax[1].set_xlabel('source epoch')
ax[0].legend(frameon=False,fontsize=6,loc='center right',bbox_to_anchor=(1,0.43)); 
plt.tight_layout(pad=0.3)
for e in ['pdf','png']: fig.savefig(f'{OUT}/task3_appendix_bandwidth.{e}',dpi=300,bbox_inches='tight')
print('ok')

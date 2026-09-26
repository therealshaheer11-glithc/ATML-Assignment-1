import os, csv, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
REPO=os.environ.get('ATML_REPO','.'); os.makedirs('report/figures/task4',exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.titlesize':8.5,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.grid':True,'grid.alpha':0.3,'pdf.fonttype':42})
H={m:list(csv.DictReader(open(f'{REPO}/task4/results/training/{m}/history.csv'))) for m in ['vanilla','gcsc','proser']}
g=lambda m,k:[float(x[k]) for x in H[m]]; e=lambda m:[int(x['epoch']) for x in H[m]]
fig,ax=plt.subplots(1,3,figsize=(7.2,2.1))
ax[0].plot(e('vanilla'),[100*v for v in g('vanilla','validation_accuracy')],color='0.35',lw=1.1,label='Vanilla')
ax[0].plot(e('gcsc'),[100*v for v in g('gcsc','validation_accuracy')],color='tab:orange',lw=1.1,label='GCSC')
ax[0].axvline(97,color='0.35',ls=':',lw=0.9); ax[0].axvline(99,color='tab:orange',ls=':',lw=0.9)
ax[0].set_ylim(60,100); ax[0].set_title('Vanilla/GCSC validation accuracy (%)'); ax[0].set_xlabel('epoch'); ax[0].legend(frameon=False,loc='lower right')
ax[1].plot(e('proser'),[100*v for v in g('proser','validation_accuracy')],'-o',ms=2,color='tab:blue',lw=1.1,label='PROSER')
ax[1].axhline(95.10,color='0.35',ls='--',lw=0.9,label='selected Vanilla')
ax[1].axvline(1,color='tab:blue',ls=':',lw=0.9)
ax[1].set_title('PROSER validation accuracy (%)'); ax[1].set_xlabel('fine-tuning epoch'); ax[1].legend(frameon=False,loc='upper right')
for k,c,l in [('known_ce','tab:blue','known CE'),('classifier_placeholder','tab:green','classifier placeholder'),('data_placeholder','tab:red','data placeholder')]:
    ax[2].plot(e('proser'),g('proser',k),color=c,lw=1.1,label=l)
ax[2].set_yscale('log'); ax[2].axvline(1,color='tab:blue',ls=':',lw=0.9)
ax[2].set_title('PROSER training losses'); ax[2].set_xlabel('fine-tuning epoch'); ax[2].legend(frameon=False,loc='upper right')
for a in ax: a.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout(pad=0.3,w_pad=0.8)
fig.savefig('report/figures/task4/task4_training_curves.pdf',bbox_inches='tight'); fig.savefig('report/figures/task4/task4_training_curves.png',dpi=200,bbox_inches='tight')

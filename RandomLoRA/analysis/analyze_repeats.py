"""Compare three completed random-mask validation runs with U0 and Full LoRA."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
random_root=root/'training/runs/random'
metrics={}
logs={}
for seed in (41,42,43):
    folder=random_root/f'seed41_mask{seed}'
    report=json.loads((folder/'validation_metrics.json').read_text())
    key='L_Random' if seed==41 else f'Random_mask{seed}'
    metrics[seed]=report[key]
    logs[seed]=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines() if x.strip()]
    assert len(logs[seed])==2000 and logs[seed][-1]['step']==2000
baseline=json.loads((root/'reference/validation_metrics.json').read_text())
u0=baseline['U0']; full=baseline['L_All']
keys=['energy_raw_MAE_eV','energy_aligned_MAE_eV','H_force_component_MAE_eV_per_A','Cu_force_component_MAE_eV_per_A']
titles=['Raw energy','Aligned energy','H force components','Cu force components']
fig,axes=plt.subplots(1,4,figsize=(14,3.7),layout='constrained')
for ax,key,title in zip(axes,keys,titles):
    vals=[u0[key],full[key]]+[metrics[s][key] for s in (41,42,43)]
    bars=ax.bar(range(5),vals,color=['#777f89','#2465a1','#d87532','#e8a554','#a84e1c'])
    ax.bar_label(bars,fmt='%.4f',padding=2,fontsize=7,rotation=0)
    ax.set_xticks(range(5),['U0','Full','R41','R42','R43'],rotation=35)
    ax.set_title(title); ax.set_ylim(0,max(vals)*1.25)
    ax.set_ylabel('MAE (eV/config)' if 'energy' in key else 'MAE (eV/Å)')
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Validation comparison · Random masks 41/42/43')
fig.savefig(root/'analysis/05_three_masks_metrics.png',dpi=200); plt.close(fig)
fig,ax=plt.subplots(figsize=(7.5,4.5),layout='constrained')
for seed,color in [(41,'#d87532'),(42,'#e8a554'),(43,'#a84e1c')]:
    vals=[r for r in logs[seed] if 'val_loss' in r]
    ax.plot([x['step'] for x in vals],[x['val_loss'] for x in vals],marker='o',label=f'Random mask {seed}',color=color)
full_log=[json.loads(x) for x in (root/'reference/training.jsonl').read_text().splitlines() if x.strip()]
vals=[r for r in full_log if 'val_loss' in r]
ax.plot([x['step'] for x in vals],[x['val_loss'] for x in vals],marker='o',color='#2465a1',label='Full LoRA')
ax.set(xlabel='Optimizer updates',ylabel='Normalized validation loss',title='Validation convergence'); ax.grid(alpha=.25); ax.legend()
fig.savefig(root/'analysis/06_three_masks_convergence.png',dpi=200); plt.close(fig)
summary={'U0':u0,'Full LoRA':full,'random_by_selection_seed':metrics,'random_mean':{k:float(np.mean([metrics[s][k] for s in metrics])) for k in keys},'random_sample_std':{k:float(np.std([metrics[s][k] for s in metrics],ddof=1)) for k in keys}}
(root/'analysis/three_masks_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

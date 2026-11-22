"""Nested per-method choice of the chain for the predicted-SMALL cells only (full cells stay on the base = first source)."""
import sys, json, numpy as np
sys.path.insert(0,'.')
from src.data import methods, folds
from src.combine2 import RT
Fo=folds(); me=methods()
names=sys.argv[1::2]; files=sys.argv[2::2]
S=np.stack([np.load(f) if f.endswith('.npy') else np.load(f)['s'] for f in files],1)
fm=lambda v:np.array([v[Fo==k].mean() for k in range(5)])
out=S[:,0].copy(); ch={}
for k in range(5):
    for m in ('pixelseal','trustmark','stegastamp'):
        c=(me==m)&(~RT); j=int(np.argmax(S[c&(Fo!=k)].mean(0))); out[c&(Fo==k)]=S[c&(Fo==k),j]; ch.setdefault(m,[]).append(names[j])
d=fm(out)-fm(S[:,0]); print(f"mélange cases small vs {names[0]}: {fm(S[:,0]).mean():.6f} -> {fm(out).mean():.6f} gain {d.mean():+.5f} plis "+" ".join(f"{x:+.4f}" for x in d)+f" {int((d>0).sum())}/5")
for m,v in ch.items(): print('   ',m,v)
fin={m:names[int(np.argmax(S[(me==m)&(~RT)].mean(0)))] for m in ('pixelseal','trustmark','stegastamp')}; print('    choix sur 510 (pour le test):',fin)
json.dump(dict(base=names[0],nested=float(fm(out).mean()),gain=float(d.mean()),fold_diffs=d.tolist(),choices=ch,final=fin),open('blend_small_report.json','w'),indent=1); np.save('blend_small_scores.npy',out)

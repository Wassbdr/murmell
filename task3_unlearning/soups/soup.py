"""Weight soup: average several state dicts of the same ViT-B/16 (strict keys, float32).
usage: python soup.py OUT.pt A.pt B.pt [C.pt ...] [--weights 0.5,0.5]
A 50/50 soup of a distilled model and its parent is what the "soup50" checkpoints are."""
import sys, torch
args = [a for a in sys.argv[1:] if not a.startswith('--')]
w = next((a.split('=', 1)[1] if '=' in a else sys.argv[sys.argv.index(a) + 1] for a in sys.argv if a.startswith('--weights')), None)
out, srcs = args[0], [a for a in args[1:] if ',' not in a]
ws = [float(x) for x in w.split(',')] if w else [1 / len(srcs)] * len(srcs)
assert len(ws) == len(srcs) and abs(sum(ws) - 1) < 1e-6
sds = [torch.load(p, map_location='cpu', weights_only=True) for p in srcs]
keys = sds[0].keys()
assert all(sd.keys() == keys for sd in sds), 'state dicts differ'
soup = {k: sum(wi * sd[k].float() for wi, sd in zip(ws, sds)) for k in keys}
assert all(torch.isfinite(v).all() for v in soup.values())
torch.save(soup, out)
print('saved', out, 'from', len(srcs), 'models, weights', ws)

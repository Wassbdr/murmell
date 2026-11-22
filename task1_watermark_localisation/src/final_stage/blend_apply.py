"""blend_apply: test masks = base candidate, except the predicted-SMALL images (RE) of the listed methods, taken from another chain.
Which chain per method is chosen on train (nested proof: blend_small.py).
  python blend_apply.py --root <task dir> --base work/test/T02_bin.npy --swap pixelseal=<A_test_bin.npy> stegastamp=<B_test_bin.npy> --out <file.npy>
"""
import argparse, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--base", required=True); ap.add_argument("--swap", nargs="*", default=[]); ap.add_argument("--out", required=True)
a = ap.parse_args(); sys.path.insert(0, a.root)
from src.combine2 import RE
from src.data import methods
me = methods("test"); B = np.load(a.base).astype(bool).copy(); assert B.shape == (1020, 128, 128)
for sw in a.swap:
    m, f = sw.split("="); A = np.load(f).astype(bool); k = (me == m) & (~RE); n = int((A[k] != B[k]).reshape(k.sum(), -1).any(1).sum()); B[k] = A[k]
    print(f"{m} small: {int(k.sum())} images taken from {f} ({n} differ from the base)")
np.save(a.out, B); print("wm frac", round(float(B.mean()), 4), "->", a.out)

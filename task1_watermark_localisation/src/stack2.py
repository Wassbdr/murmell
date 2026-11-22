"""STK: 128x128 stacker used as a FOLD-DEPENDENT candidate source inside src.full_nested (--stack 1). Human spec 23:45.
One small conv net for the three methods fuses, per image:
  * mean of the fold-clean FULL-regime synthetic maps (runs with fold_excluded, regime full: *_xfall)      [X]
  * mean of the fold-clean SMALL-regime synthetic maps (S1_*_xfall, T1_*_xfall ... when they exist)        [XS]
  * real fine-tuned maps, out-of-fold (FT_*: full-base and small-base separately)                           [FTf, FTs]
  * edited-zone detector, out-of-fold: watermark channel and 1 - edited zone                                [E1w, E1n]
  * decoder-ranked box mask (GT-free rule) + "has box" plane, method one-hot, predicted regime
STRICT inputs only: every map comes from a model that saw neither the image nor any shape of its fold (or uses no
label at all).  Non-fold-clean runs (plain F*/S*/T*/D*, teammate's snapDF whose folds differ) are NOT inputs.
A missing group for a method is a constant 0.5 plane (the method one-hot tells the net).
Nesting (15 fits, fixed hyper-parameters of src/stack.py written at 19:55, never tuned):
  M(-k)    trained on folds != k        -> predicts fold k and the test set
  M(-k,-j) trained on folds not in {k,j} -> predicts folds k and j
  train[k]  (selection maps when k is the OUTER fold): fold k from M(-k), fold j != k from M(-k,-j)  => nothing about
            fold k is used for any choice scored on fold k
  train["all"] = standard out-of-fold map (fold j from M(-j)), used for the final choice applied to the test set
  test = mean_k M(-k)(test features)
"""
import json
import re
import time

import numpy as np
import torch

from src.combine2 import RE, RT
from src.data import METHODS, ROOT, folds, masks128, methods
from src.stack import Net, loss_fn


def groups():
    """group name -> method -> list of run names (complete, fold-clean)."""
    out = {}
    for d in sorted((ROOT / "work/runs").iterdir()):
        n = d.name
        if not all((d / f).exists() and (d / f).stat().st_size > 0 for f in ("train.npy", "test.npy", "args.json")):
            continue
        if re.search(r"_xf\d$|_f\d$|smoke|_tr$", n):
            continue
        a = json.loads((d / "args.json").read_text())
        if not a.get("fold_excluded") or "method" not in a:
            continue
        if a.get("kind") == "edit_detector":
            g = n[:3]                                         # E1w / E1n
        elif "bs_real" in a:                                  # any src.finetune_real run (FT_, FTt_, FTx_, FTxt_)
            if not a.get("base_fold_clean"):
                continue                                      # base saw the shapes of all folds: not a strict stacker input
            g = "FTs" if a.get("regime", "full").startswith("small") else "FTf"
        elif a.get("kind") == "grad_evidence":
            continue
        else:
            g = "XS" if a.get("regime", "full").startswith("small") else "X"
        out.setdefault(g, {}).setdefault(a["method"], []).append(n)
    return out


def features(ovr, ovt, verbose=True):
    gr = groups(); names_ = sorted(gr); me, mt = methods(), methods("test")
    C = len(names_) + 6
    Xtr = np.zeros((510, C, 128, 128), np.float32); Xte = np.zeros((1020, C, 128, 128), np.float32)
    for c, g in enumerate(names_):
        Xtr[:, c] = 0.5; Xte[:, c] = 0.5
        for m in METHODS:
            rs = gr[g].get(m, [])
            if rs:
                Xtr[me == m, c] = np.mean([np.load(ROOT / "work/runs" / r / "train.npy").astype(np.float32) for r in rs], 0)
                Xte[mt == m, c] = np.mean([np.load(ROOT / "work/runs" / r / "test.npy").astype(np.float32) for r in rs], 0)
    c = len(names_)
    for X, ov, meth, reg in ((Xtr, ovr, me, RT), (Xte, ovt, mt, RE)):
        for i, bm in ov.items():
            X[i, c] = bm; X[i, c + 1] = 1.0
        for j, m in enumerate(METHODS):
            X[meth == m, c + 2 + j] = 1.0
        X[:, c + 5] = reg[:, None, None].astype(np.float32)
    if verbose:
        print("STACK channels:", {g: {m[:2]: gr[g].get(m, []) for m in METHODS} for g in names_}, "+ box, has_box, method x3, regime", flush=True)
    return Xtr, Xte, names_


def fit_predict(Xtr, Ytr, evs, dev, seed=0, epochs=60):
    torch.manual_seed(seed); net = Net(c=Xtr.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr[:, None].astype(np.float32)).to(dev)
    n = len(xt); steps = epochs * ((n + 15) // 16); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for j in range(0, n, 16):
            b = perm[j:j + 16]; xb, yb = xt[b], yt[b]
            if torch.rand(1, generator=g).item() < 0.5:
                xb, yb = xb.flip(-1), yb.flip(-1)
            loss = loss_fn(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval(); outs = []
    with torch.no_grad():
        for X in evs:
            outs.append(np.concatenate([torch.sigmoid(net(torch.from_numpy(X[s:s + 128]).to(dev))).cpu().numpy()[:, 0]
                                        for s in range(0, len(X), 128)]))
    return outs


def stack_maps(ovr, ovt, dev=None, epochs=60):
    dev = dev or ("cuda:0" if torch.cuda.is_available() else "cpu")
    import os
    epochs = int(os.environ.get("WMLOC_STACK_EPOCHS", epochs))          # smoke tests only; 60 is the fixed setting
    Xtr, Xte, names_ = features(ovr, ovt); G, Fo = masks128().astype(np.float32), folds()
    if not names_:
        print("STACK: no fold-clean map group available -> skipped", flush=True); return None
    t0 = time.time(); F = [np.where(Fo == k)[0] for k in range(5)]
    train = {k: np.zeros((510, 128, 128), np.float32) for k in list(range(5)) + ["all"]}; test = np.zeros((1020, 128, 128), np.float32)
    for k in range(5):
        tr = np.where(Fo != k)[0]
        pk, pt = fit_predict(Xtr[tr], G[tr], [Xtr[F[k]], Xte], dev, seed=k, epochs=epochs)
        train[k][F[k]] = pk; train["all"][F[k]] = pk; test += pt / 5
        for j in range(k + 1, 5):
            tr2 = np.where((Fo != k) & (Fo != j))[0]
            pkk, pjj = fit_predict(Xtr[tr2], G[tr2], [Xtr[F[k]], Xtr[F[j]]], dev, seed=10 * k + j, epochs=epochs)
            train[j][F[k]] = pkk; train[k][F[j]] = pjj
        print(f"STACK outer fold {k} done {time.time() - t0:.0f}s", flush=True)
    return {"train": train, "test": test, "channels": names_}

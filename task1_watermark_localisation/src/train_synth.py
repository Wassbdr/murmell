"""Task-1 pipeline on the synthetic visible-watermark set, end to end on CPU.

Same recipe as src/train_full.py (U-Net on RGB + high-pass input, BCE + soft-mIoU loss, logits pooled to cells, trained
ONLY on a synthetic stream) and the same evaluation as src/eval_cv.py (threshold per group chosen on the 4 other folds,
scored on the fifth, official metric of src/metric.py). The evaluation set uses another seed than the training stream.

    python -m src.train_synth --steps 1500                 # ~30 min on 8 CPU cores
Outputs: weights/<name>.pt (float16 state dict) + weights/<name>.json (card: config, metrics, threshold, SHA-256),
         work/synth/oof.npy (eval probabilities), work/synth/results.json and results/<name>_results.json (committed).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import ROOT
from src.eval_cv import evaluate
from src.metric import miou_batch
from src.synth_visible import make_item, make_set
from src.train_full import Net, loss_fn

KINDS_EVAL = ("text", "logo", "tiled", "clean")
CFG = dict(encoder="mobilenet_v2", size=256, cells=128, decoder_channels=[128, 64, 32, 16, 16])


def build_net(cfg=CFG, weights="imagenet"):
    return Net(cfg["encoder"], pool=cfg["size"] // cfg["cells"], weights=weights,
               decoder_channels=tuple(cfg["decoder_channels"]))


class Stream(Dataset):
    def __init__(self, n, seed, size, cells):
        self.n, self.seed, self.size, self.cells = n, seed, size, cells

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x, m, _ = make_item(self.seed, i, self.size, self.cells)
        return torch.from_numpy(np.array(x)).permute(2, 0, 1), torch.from_numpy(m.astype(np.float32))[None]


@torch.no_grad()
def predict_batch(net, X, bs=32):
    net.eval(); out = []
    for s in range(0, len(X), bs):
        x = torch.from_numpy(np.ascontiguousarray(X[s:s + bs])).permute(0, 3, 1, 2)
        out.append(torch.sigmoid(net(x))[:, 0].numpy())
    net.train()
    return np.concatenate(out)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_eval", type=int, default=500); ap.add_argument("--eval_seed", type=int, default=1)
    ap.add_argument("--n_cal", type=int, default=200); ap.add_argument("--cal_seed", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--name", default="wm_unet_synth"); ap.add_argument("--weights", default="imagenet")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    work = ROOT / "work" / "synth"; work.mkdir(parents=True, exist_ok=True)
    wdir = ROOT / "weights"; wdir.mkdir(exist_ok=True)
    train_seed = 1000 + a.seed
    assert train_seed not in (a.eval_seed, a.cal_seed)
    ds = Stream(a.steps * a.bs, train_seed, CFG["size"], CFG["cells"])
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, shuffle=False, drop_last=True,
                    persistent_workers=a.workers > 0)
    net = build_net(weights=None if a.weights == "none" else a.weights)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=max(0.05, 2 / a.steps))
    t0 = time.time(); run = None
    for step, (x, y) in enumerate(dl, 1):
        loss = loss_fn(net(x), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 50 == 0 or step == a.steps:
            print(f"step {step}/{a.steps} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
    train_s = time.time() - t0
    net.eval()

    # calibration set (own seed): the single threshold used by predict.py / the app (kind unknown at inference)
    Xc, Mc, _ = make_set(a.n_cal, a.cal_seed, CFG["size"], CFG["cells"])
    Pc = predict_batch(net, Xc)
    thrs = np.round(np.linspace(0.1, 0.9, 17), 2)
    cal = [float(miou_batch(Pc >= t, Mc).mean()) for t in thrs]
    thr = float(thrs[int(np.argmax(cal))])

    # evaluation set (another seed): per-fold mIoU, thresholds chosen on the 4 other folds
    X, M, meta = make_set(a.n_eval, a.eval_seed, CFG["size"], CFG["cells"])
    t1 = time.time(); P = predict_batch(net, X); infer_ms = (time.time() - t1) / len(X) * 1000
    np.save(work / "oof.npy", P.astype(np.float16))
    kind = np.array([m["kind"] for m in meta]); fold = np.array([m["fold"] for m in meta])
    res_k, per_img = evaluate(P, G=M, F=fold, me=kind, rg=kind == "tiled", kinds=KINDS_EVAL, name="nested per-kind thr")
    res_g, per_img_g = evaluate(P, groups=np.array(["all"] * len(P)), G=M, F=fold, me=kind, rg=kind == "tiled",
                                kinds=KINDS_EVAL, name="nested single thr")
    fixed = miou_batch(P >= thr, M)
    op = np.array([m["opacity"] for m in meta])
    bins = [(0.15, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.91)]
    by_op = {f"{lo:.2f}-{hi:.2f}": float(fixed[(op >= lo) & (op < hi)].mean()) for lo, hi in bins}
    zeros = float(miou_batch(np.zeros_like(M), M).mean())
    results = dict(
        config=dict(CFG, steps=a.steps, bs=a.bs, lr=a.lr, seed=a.seed, train_seed=train_seed, eval_seed=a.eval_seed,
                    n_eval=a.n_eval, cal_seed=a.cal_seed, n_cal=a.n_cal, init=a.weights,
                    threads=torch.get_num_threads()),
        miou_nested_per_kind=res_k["all"][0], miou_nested_per_kind_std=res_k["all"][1], fold_miou=res_k["fold"],
        thresholds_per_kind=res_k["thr"], per_kind={k: res_k[k][0] for k in KINDS_EVAL},
        miou_nested_single_thr=res_g["all"][0], fold_miou_single_thr=res_g["fold"],
        miou_at_cal_thr=float(fixed.mean()), cal_threshold=thr, miou_by_opacity=by_op, baseline_all_zeros=zeros,
        per_image_iou=[round(float(v), 4) for v in fixed], train_seconds=round(train_s), infer_ms_per_image=round(infer_ms, 1),
    )
    (work / "results.json").write_text(json.dumps(results, indent=1))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{a.name}_results.json").write_text(json.dumps(results, indent=1))
    sd = {k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()}
    wpath = wdir / f"{a.name}.pt"; torch.save(sd, wpath)
    card = dict(name=a.name, file=wpath.name, sha256=sha256(wpath), config=results["config"], threshold=thr,
                metrics={k: results[k] for k in ("miou_nested_per_kind", "fold_miou", "miou_nested_single_thr",
                                                 "miou_at_cal_thr", "per_kind", "miou_by_opacity", "baseline_all_zeros")},
                hardware=f"CPU, {torch.get_num_threads()} threads", train_seconds=results["train_seconds"],
                rebuild="python -m src.train_synth " + " ".join(f"--{k} {v}" for k, v in vars(a).items()))
    (wdir / f"{a.name}.json").write_text(json.dumps(card, indent=1))
    print(json.dumps({k: card["metrics"][k] for k in ("miou_nested_per_kind", "fold_miou", "miou_at_cal_thr")}, indent=1))
    print(f"weights {wpath} ({wpath.stat().st_size / 2**20:.1f} MB) sha256 {card['sha256'][:16]}… train {train_s:.0f}s")


if __name__ == "__main__":
    main()

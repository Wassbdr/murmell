"""SAM snapping driven by the method's ENSEMBLE map (Sun 07:05).  Small regime only: a transplanted object is ONE SAM segment
(in-sample look on 383 train images: the best segment reaches mIoU 0.958 on PixelSeal object images, our maps 0.70-0.78),
whereas rectangles and full-regime holes do not follow SAM segments (snapping hurts there).
For each method:
  E      = mean of ALL runs usable in small cells (run_regime in any / small; list frozen in work/sam/ens_members.json;
           no score is used to build it)                                              -> run <ENS>            (the plain mean)
  s*, J  = per image, the SAM segment (<= 50 % of the frame) with the best soft IoU against E, and that soft IoU
  <base>_samE   : base map snapped to s* (0.25 p outside, 0.5 + 0.5 mean_s*(p) inside)   - always
  <base>_samR30 : snapped only when J >= 0.30 (label-free router object / rectangle), else the base map
  <base>_samR40 : same with J >= 0.40
All new runs get regime "small" (they use maps of small-trained runs): src.full_nested may use them in S cells only, and
chooses between them, their bases and the thresholds in the inner folds like for any other run.  No label is used here.
    python -m src.sam_ens            # writes the runs (run dirs are pre-created placeholders)
"""
import json

import numpy as np

from src.data import METHODS, ROOT, methods
from src.sam_refine import load_segs

ENS = {"pixelseal": "DS9_ps_ens", "trustmark": "DS9_tm_ens", "stegastamp": "S9_ss_ens"}
BASES = {"pixelseal": ["FTxt_R_DT2_ps_amp4", "RO_D5_ps_noamp_xfall_tta", "FTt_DT5_ps_amp4", "D5_ps_noamp_xfall_tta", "FTt_R_DT2_ps_amp4"],
         "trustmark": ["D2_tm_xfall_tta", "FTt_R_DT2_tm", "RF_D2_tm_xfall_tta", "FTt_DT5_tm", "DT2_tm_xfall_tta"],
         "stegastamp": ["FTt_R_S1_ss", "R_F3_ss_xfall_tta", "F3_ss_xfall_tta", "FT_S1_ss", "FTt_S5_ss"]}
VARIANTS = {"samE": 0.0, "samR30": 0.30, "samR40": 0.40}
MEMBERS = ROOT / "work/sam/ens_members.json"
import os
if os.environ.get("SAMENS", "") == "RS":      # Sun 09:40: the same chain on the runs trained with SAM-silhouette object items (RS_*), fixed a-priori lists
    ENS = {"pixelseal": "DS8_ps_ens", "trustmark": "DS8_tm_ens", "stegastamp": "S8_ss_ens"}
    BASES = {"pixelseal": ["RS_D5_ps_noamp_xfall_tta", "RS_R_DT2_ps_amp4_tta", "RS_DT5_ps_amp4_tta"],
             "trustmark": ["RS_D2_tm_xfall_tta", "RS_R_DT2_tm_tta", "RS_DT2_tm_tta"],
             "stegastamp": ["RS_R_F3_ss_xfall_tta", "RS_S5_ss_tta", "RS_R_S1_ss_tta"]}
    MEMBERS = ROOT / "work/sam/ens_members_RS.json"
    if not MEMBERS.exists():
        MEMBERS.write_text(json.dumps({"pixelseal": ["RS_DT5_ps_amp4_tta", "RS_DT5_ps_amp4_s1_tta", "RS_R_DT2_ps_amp4_tta", "RS_D5_ps_noamp_xfall_tta"],
                                       "trustmark": ["RS_DT2_tm_tta", "RS_R_DT2_tm_tta", "RS_DT5_tm_tta", "RS_D2_tm_xfall_tta"],
                                       "stegastamp": ["RS_R_S1_ss_tta", "RS_S5_ss_tta", "RS_R_F3_ss_xfall_tta"]}, indent=1))


def members():
    if MEMBERS.exists() and MEMBERS.stat().st_size:
        return json.loads(MEMBERS.read_text())
    from src.nested_cv import run_regime
    from src.night import runs
    out = {m: [] for m in METHODS}
    for r in runs():
        if r == "snapDF" or "_sam" in r or r.endswith("_ens"):
            continue
        a = json.loads((ROOT / "work/runs" / r / "args.json").read_text())
        if run_regime(r) in ("any", "small", "small_proc"):
            out[a["method"]].append(r)
    with open(MEMBERS, "w") as f:
        f.write(json.dumps(out, indent=1))
    return out


def best_seg(p, segs):
    tot = float(p.sum()); best, bs = 0.0, None
    for s in segs:
        n = float(s.sum())
        if n > 0.5 * 16384:
            continue
        i_ = float(p[s].sum()); j_ = i_ / (n + tot - i_ + 1e-6)
        if j_ > best:
            best, bs = j_, s
    return best, bs


def snap(p, s):
    q = 0.25 * p; q[s] = 0.5 + 0.5 * float(p[s].mean())
    return q


def save(name, tr, te, args):
    o = ROOT / "work/runs" / name; o.mkdir(parents=True, exist_ok=True)
    with open(o / "train.npy", "wb") as f:
        np.save(f, tr.astype(np.float16))
    with open(o / "test.npy", "wb") as f:
        np.save(f, te.astype(np.float16))
    with open(o / "args.json", "w") as f:             # written last: src.night ignores a run whose args.json is empty
        f.write(json.dumps(args))
    print("WROTE", name, flush=True)


def main():
    M = members(); S = {"train": load_segs("train"), "test": load_segs("test")}; ids = {"train": methods(), "test": methods("test")}
    for m in METHODS:
        idx = {sp: np.where(ids[sp] == m)[0] for sp in ("train", "test")}
        E = {sp: np.mean([np.load(ROOT / "work/runs" / r / f"{sp}.npy").astype(np.float32) for r in M[m]], 0) for sp in ("train", "test")}
        pick = {sp: [best_seg(E[sp][k], S[sp].get(int(i), [])) for k, i in enumerate(idx[sp])] for sp in ("train", "test")}
        J = {sp: np.array([j for j, _ in pick[sp]]) for sp in pick}
        print(f"{m}: ensemble of {len(M[m])} runs | J >= .3: train {np.mean(J['train'] >= .3):.3f} test {np.mean(J['test'] >= .3):.3f} | J >= .4: train {np.mean(J['train'] >= .4):.3f} test {np.mean(J['test'] >= .4):.3f}", flush=True)
        base_args = dict(method=m, regime="small", sam_ens=M[m], sam_segments="work/sam (SAM ViT-H automatic masks)")
        save(ENS[m], E["train"], E["test"], dict(base_args, kind="ensemble mean of the S-usable runs"))
        for b in BASES[m] + [ENS[m]]:
            P = E if b == ENS[m] else {sp: np.load(ROOT / "work/runs" / b / f"{sp}.npy").astype(np.float32) for sp in ("train", "test")}
            for v, tau in VARIANTS.items():
                out = {sp: np.stack([snap(P[sp][k], s) if (s is not None and j >= tau) else P[sp][k] for k, (j, s) in enumerate(pick[sp])])
                       for sp in ("train", "test")}
                save(f"{b}_{v}", out["train"], out["test"], dict(base_args, sam_base=b, sam_variant=v, sam_tau=tau))


if __name__ == "__main__":
    main()

"""Night pass (human 20:50, item 6): full_nested on every FINISHED run + every COMPLETE box set, write the candidate,
print its nested CV vs the best sent.  Sending stays with autosubmit (same rule: clean nested CV > best sent, checker).
    python -m src.night --name N2300            # -> work/nested/N2300.json, submission + work/archive/N2300/
    python -m src.night --name N2300 --dry 1    # only list what would be used
Excluded runs: train-only-background runs (*_tr, decision 20:37), per-fold pieces (*_xf[0-4]; *_xfall is used),
smoke runs, runs without both train.npy and test.npy.  Box sets: <method>_<split>_<tag>_<k>of<n>.json with all n shards
present for BOTH splits.
"""
import argparse
import glob
import json
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np

from src.data import ROOT


def runs():
    out = []
    for d in sorted((ROOT / "work/runs").iterdir()):
        n = d.name
        if n.endswith("_tr") or "_tr_" in n or re.search(r"_xf\d$", n) or "smoke" in n or re.search(r"_f\d$", n):   # _f<k> = per-fold pieces (E1)
            continue
        if not all((d / f).exists() and (d / f).stat().st_size > 0 for f in ("train.npy", "test.npy", "args.json")):
            continue
        if not np.isfinite(np.load(d / "train.npy", mmap_mode="r")[:, 0, 0].astype(np.float32)).all():
            continue
        if "PL_" in n or "pseudo_ft" in (d / "args.json").read_text():
            continue        # trained on TEST images: human decision 05:50 = NO (no switch, pruned or not)
        out.append(n)
    if __import__("os").environ.get("WMLOC_PRUNE", "1") == "1":
        out = prune(out)
    return out + ["snapDF"]


def prune(runs_):
    """A-PRIORI pool pruning (no score involved), to limit selection noise as the pool grows (04:00, M05 < M04 with +9 runs):
    1. a run whose shift-TTA twin <run>_tta exists is dropped (the twin is the same model, better predicted);
    2. PixelSeal / TrustMark: only the decoder-initialised family is kept (D*, DT*, DS* and their R_/FT*/xfall versions);
       the ImageNet-initialised generation (F*, S*, T1*), the edited-zone detector (E1*) and S3* are dropped;
    3. StegaStamp (no decoder weights available): everything is kept except E1* and S3*."""
    keep = []; S = set(runs_)
    for n in runs_:
        m = json.loads((ROOT / "work/runs" / n / "args.json").read_text()).get("method")
        if n + "_tta" in S or re.match(r"(E1|S3)", n):
            continue
        if "PL_" in n:
            continue        # PL_* = self-training on the unlabeled TEST images: refused by the human (05:50)
        if m in ("pixelseal", "trustmark") and not re.match(r"(FT_|FTx_|FTt_|FTxt_|R[A-Z0-9]*_)*(D\d|DT\d|DS\d)", n):
            continue
        keep.append(n)
    return keep


def box_sets():
    have = defaultdict(set); nsh = {}
    for f in glob.glob(str(ROOT / "work/boxes/*.json")):
        if __import__("os").path.getsize(f) == 0:
            continue
        m = re.match(r"(pixelseal|trustmark|stegastamp)_(train|test)_(.+)_(\d+)of(\d+)\.json$", f.split("/")[-1])
        if m:
            meth, sp, tag, k, n = m.groups(); have[(meth, tag, sp)].add(int(k)); nsh[(meth, tag)] = int(n)
    out = []
    DROP = {("trustmark", "loc09"), ("trustmark", "S1_tm")}   # loc09: no new coverage (21:14); S1_tm superseded by S1_tm_grid
    for (meth, tag), n in sorted(nsh.items()):
        if (meth, tag) in DROP:
            continue
        if all(have[(meth, tag, sp)] == set(range(n)) for sp in ("train", "test")):
            out.append(f"{meth}:{tag}")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="N" + __import__("time").strftime("%H")); ap.add_argument("--dry", type=int, default=0)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--extra_boxes", nargs="*", default=["pixelseal:S1_ps_amp4_grid"])   # partial but GT-free & same rule on train/test (shards 2-3)
    a = ap.parse_args()
    sent = {e["name"] for e in json.loads((ROOT / "work/submit_log.json").read_text())}
    for n in list(sent):      # the base of a SENT selector candidate is part of its record: never overwrite it either
        sp = ROOT / "work/archive" / n / "spec.json"
        if sp.exists() and sp.stat().st_size:
            b = json.loads(sp.read_text()).get("base")
            if b and not str(b).startswith("work/"):
                sent.add(b)
    if a.name in sent:      # never overwrite a sent archive: fall back to the pre-created M<HH> / next free M name
        alt = [f"M{h:02d}" for h in list(range(int(__import__("time").strftime("%H")), 10)) + list(range(2, 26))]
        alt = [n for n in alt if n not in sent and (ROOT / "work/archive" / n).exists()]
        if not alt:
            print(f"NIGHT {a.name}: name already SENT and no free M<HH> placeholder -> keep"); return
        print(f"NIGHT: {a.name} already sent -> using {alt[0]}"); a.name = alt[0]
    R, B = runs(), box_sets()
    B = B + [b for b in a.extra_boxes if b not in B]
    print("runs", len(R), " ".join(R)); print("boxes", " ".join(B))
    import os, socket
    st = os.environ.get("WMLOC_STACK", "auto")     # stacker: on GPU machines that are not shared login nodes (15 small fits)
    if st == "auto":
        import torch
        st = "1" if torch.cuda.is_available() and not socket.gethostname().startswith("jrlogin") else "0"
    cmd = [sys.executable, "-m", "src.full_nested", "--name", a.name, "--cands", *R, "--boxes", *B, "--topk", str(a.topk),
           "--procs", os.environ.get("WMLOC_PROCS", "12"), "--write", "1", "--stack", st]
    print("CMD", " ".join(cmd[1:]))
    if a.dry:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)
    log = json.loads((ROOT / "work/submit_log.json").read_text())
    from tools.autosubmit import is_clean          # human 05:50: the reference is the best CLEAN send
    best = max(json.loads((ROOT / "work/nested" / f"{e['name']}.json").read_text())["cv_foldmean"] for e in log
               if is_clean(e["name"]))
    cv = json.loads((ROOT / "work/nested" / f"{a.name}.json").read_text())["cv_foldmean"]
    print(f"NIGHT {a.name}: nested {cv:.6f} vs best sent {best:.6f} -> {'SEND' if cv > best else 'keep'}")
    # gated last step: an extra candidate S<HH> exists only if the
    # selector gains >= 0.003 nested AND >= 4/5 outer folds on THIS base (human rule); WMLOC_SEL=0 disables it
    if os.environ.get("WMLOC_SEL", "1") == "1":
        try:
            import lightgbm, sklearn      # noqa: F401
        except ImportError:
            print("SEL skipped (lightgbm / scikit-learn not installed)"); return
        import time
        sname = next((n for n in [f"S{time.strftime('%H')}"] + [f"S{h:02d}" for h in range(3, 31)]
                      if n not in sent and (ROOT / "work/archive" / n).exists()), None)
        if sname:
            r = subprocess.run([sys.executable, "-m", os.environ.get("WMLOC_SELMOD", "src.selector3"), "--base", a.name, "--cand", sname, "--procs", os.environ.get("WMLOC_PROCS", "24")],
                               cwd=ROOT, capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith("SEL gate")]
            print(line[-1] if line else "SEL failed: " + r.stderr[-300:])
            if line and "KEEP" in line[-1]:
                cv2 = json.loads((ROOT / "work/nested" / f"{sname}.json").read_text())["cv_foldmean"]
                print(f"NIGHT {sname}: nested {cv2:.6f} vs best sent {best:.6f} -> {'SEND' if cv2 > best else 'keep'}")


if __name__ == "__main__":
    main()

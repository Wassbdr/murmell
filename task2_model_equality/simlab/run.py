"""Task 2 on a seeded synthetic bench: 33 toy endpoints (3 families x 11), the pipeline's statistics and its own decision rules.

    python -m simlab.run --seeds 0 1 2 3 4 --out out/simlab          (from task2_model_equality/, CPU, ~1 min per seed)
Writes out/simlab/partition_<seed>.csv and truth_<seed>.csv; results/simlab.json and results/simlab_family_comparison.png.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import stats
from .toy import build_bench, collect

HERE = Path(__file__).resolve().parents[1]


def run_seed(seed, out):
    eps, bases = build_bench(seed)
    answers = {e.name: collect(e) for e in eps}
    fam_of, ll, st = stats.measure(eps, answers, bases, seed)
    dec = {}; gfp = {}
    for f, st_f in st.items():
        if not st_f:
            continue
        gfp[f] = stats.fingerprint(sorted(st_f), answers, bases[f], st_f, seed=seed)
        d, _ = stats.decide_family(st_f, gfp[f]); dec.update({e: dict(v, family=f) for e, v in d.items()})
    part = {e: dict(family=v["family"], model=v["model"]) for e, v in dec.items()}
    with open(out / f"truth_{seed}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["endpoint", "family", "model", "kind"])
        for e in eps:
            w.writerow([e.name, e.family, e.model, e.role])
    with open(out / f"partition_{seed}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["endpoint", "family", "model", "kind"])
        for e in sorted(part):
            w.writerow([e, part[e]["family"], part[e]["model"], dec[e]["kind"]])
    sc = stats.mp.score_truth(part, out / f"truth_{seed}.csv")
    roles = {e.name: e.role for e in eps}
    errors = [dict(endpoint=e.name, role=e.role, predicted=dec[e.name]["kind"], why=dec[e.name]["why"][:200])
              for e in eps if (dec[e.name]["kind"] == "base") != (e.model == "base")]
    selfz = {e: gfp[fam_of[e]]["self"][e]["z"] for e in dec}
    return dict(seed=seed, score=sc, errors=errors, family_ll=ll, family_true={e.name: e.family for e in eps},
                roles=roles, self_z=selfz, kinds={e: dec[e]["kind"] for e in dec})


def chart(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = sorted(r["roles"], key=lambda e: (r["family_true"][e], r["roles"][e]))
    fams = sorted({f for v in r["family_ll"].values() for f in v})
    M = np.array([[r["family_ll"][e][f] for f in fams] for e in order]); M = M - M.max(1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw=dict(width_ratios=[1, 1.6]))
    im = ax[0].imshow(M.T, aspect="auto", cmap="viridis", vmin=-1.0, vmax=0)
    ax[0].set_yticks(range(len(fams)), [f"base {f}" for f in fams]); ax[0].set_xticks(range(len(order)), order, rotation=90, fontsize=7)
    ax[0].set_title("Stage a: log-likelihood per token under each candidate base\n(relative to the best candidate; 0 = chosen)")
    fig.colorbar(im, ax=ax[0], orientation="horizontal", pad=0.25, fraction=0.05)
    col = {"plain": "#4c72b0", "system": "#64b5cd", "persona": "#64b5cd", "hot": "#8172b2", "topk": "#8172b2",
           "watermark": "#55a868", "fp8": "#c44e52"}
    zs = [r["self_z"][e] for e in order]; cs = [col.get(r["roles"][e], "#dd8452") for e in order]
    ax[1].bar(range(len(order)), zs, color=cs)
    ax[1].set_xticks(range(len(order)), [f"{e} {r['roles'][e]}" for e in order], rotation=90, fontsize=7)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_ylabel("gradient fingerprint self z")
    for i, e in enumerate(order):
        ax[1].text(i, max(zs[i], 0) + 0.5, r["kinds"][e][:4], ha="center", fontsize=6, rotation=90)
    ax[1].set_title(f"Fingerprint per endpoint (orange = LoRA doses, red = fp8); label = decision\n"
                    f"seed {r['seed']}: family MCC {r['score']['mcc_family']}, model MCC {r['score']['mcc_model']}, score {r['score']['score']}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="out/simlab"); a = ap.parse_args()
    out = Path(a.out); out = out if out.is_absolute() else HERE / out; out.mkdir(parents=True, exist_ok=True)
    res = []
    for s in a.seeds:
        t = time.time(); r = run_seed(s, out); res.append(r)
        print(f"seed {s}: {r['score']}  errors {[(x['endpoint'], x['role'], x['predicted']) for x in r['errors']]}  {time.time() - t:.0f}s", flush=True)
    rdir = HERE / "results"; rdir.mkdir(exist_ok=True)
    chart(res[0], rdir / "simlab_family_comparison.png")
    summ = dict(seeds=a.seeds, score_mean=float(np.mean([r["score"]["score"] for r in res])),
                mcc_family_mean=float(np.mean([r["score"]["mcc_family"] for r in res])),
                mcc_model_mean=float(np.mean([r["score"]["mcc_model"] for r in res])),
                per_seed=[dict(seed=r["seed"], **r["score"], errors=r["errors"]) for r in res])
    (rdir / "simlab.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in ("score_mean", "mcc_family_mean", "mcc_model_mean")}))


if __name__ == "__main__":
    main()

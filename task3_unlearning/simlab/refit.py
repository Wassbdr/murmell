"""Refit loop + recorded forecasts on the toy task-3 world, then the repository's own backtest on them.

Each round, from earlier receipts only: calibrate our offline simulator against the official scores (slope of public
score on simulated score), move the global parameters of the best model so far by a bounded step that maximises the
forecast, FORECAST the candidate as parent score + slope x (simulated gain over the parent), then 'submit' it (toy
public score). Forecasts and receipts are written in the format read by backtest/backtest_recorded_forecasts.py,
which scores them unchanged.

    python -m simlab.refit --seed 0 --rounds 14         (from task3_unlearning/, CPU, a few seconds)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .score import HI, LO, PARAMS, World

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "backtest"))
import backtest_recorded_forecasts as bt  # noqa: E402


def slope(sim, actual, lo=0.3, hi=2.0):
    """least-squares slope of the official score on the simulated score over earlier receipts (bounded)."""
    x = np.asarray(sim) - np.mean(sim); y = np.asarray(actual) - np.mean(actual)
    return float(np.clip((x @ y) / max(x @ x, 1e-12), lo, hi))


def forecast(parent_score, sim_parent, sim_cand, b):
    return parent_score + b * (sim_cand - sim_parent)


def propose(sim, parent, rng, step=0.08, n=400):
    """bounded local step: best simulated score among random moves of at most `step` (relative to the range) per parameter."""
    cand = np.clip(parent + rng.uniform(-step, step, (n, len(parent))) * (HI - LO), LO, HI)
    v = np.array([sim.components(c)["score"] for c in cand])
    return cand[int(np.argmax(v))]


def run(seed=0, rounds=14, n_init=8, out=None):
    world = World(seed); sim = World(seed, simulator=True); rng = np.random.default_rng(seed + 100)
    simscore = lambda th: sim.components(np.asarray(th))["score"]
    out = Path(out or HERE / "out" / f"simlab_{seed}"); out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("receipt-*.json"):
        p.unlink()
    start = np.array([1.2, 0.8, 1.0, 0.5])
    hist = []; sid = 1000; jobs = {}

    def submit(name, theta):
        nonlocal sid
        sid += int(rng.integers(3, 30))                                   # receipt ids grow, other teams submit in between
        s = world.public_score(theta)
        (out / f"receipt-{sid}.json").write_text(json.dumps(dict(status="success", model=f"{name}.pt", submission_id=sid,
                                                                 score=round(s, 4))))
        hist.append(dict(id=sid, name=name, theta=theta.tolist(), score=round(s, 4)))
        return s

    for i in range(n_init):                                               # first designs around the start point
        submit(f"init_{i}", np.clip(start + rng.normal(0, 0.15, 4) * (HI - LO), LO, HI))
    for _ in range(rounds):
        sc = np.array([h["score"] for h in hist]); best = hist[int(np.argmax(sc))]; parent = np.array(best["theta"])
        b = slope([simscore(h["theta"]) for h in hist], sc)
        cand = propose(sim, parent, rng)
        basis = hist[-1]["id"]; fc = forecast(best["score"], simscore(parent), simscore(cand), b)
        jobs[str(basis)] = dict(parent=best["name"], parent_score=best["score"], forecast=round(fc, 6), slope=round(b, 3),
                                theta=dict(zip(PARAMS, np.round(cand, 4).tolist())))
        submit(f"codex_prob_recu_{basis}", cand)
    (out / "iterations_recus.json").write_text(json.dumps(jobs, indent=1))
    report = bt.build_report(out)
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    best = max(hist, key=lambda h: h["score"])
    report["toy"] = dict(seed=seed, best_public=best["score"], best_full=world.components(np.array(best["theta"]))["score"],
                         best_theta=dict(zip(PARAMS, best["theta"])))
    return report, hist


def chart(report, hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["rows"]; s = report["summary"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax[0].plot(x, [r["actual"] for r in rows], "o-", color="k", label="actual (later receipt)")
    ax[0].plot(x, [r["forecast"] for r in rows], "s--", color="#dd8452", label="forecast (earlier receipts only)")
    ax[0].plot(x, [r["parent_score"] for r in rows], "^:", color="#8c8c8c", label="parent baseline")
    ax[0].set_xticks(x, [str(r["receipt_id"]) for r in rows], rotation=60, fontsize=7); ax[0].set_xlabel("receipt id")
    ax[0].set_ylabel("public score"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Forecast vs actual: MAE {s['forecast_mae']:.4f} vs parent {s['parent_mae']:.4f}, "
                    f"direction {s['direction_accuracy']:.0%}")
    a = [r["actual"] for r in rows]; f = [r["forecast"] for r in rows]
    lo, hi = min(a + f) - 0.01, max(a + f) + 0.01
    ax[1].plot([lo, hi], [lo, hi], color="#8c8c8c", lw=0.8); ax[1].scatter(f, a, color="#dd8452")
    ax[1].set_xlabel("forecast"); ax[1].set_ylabel("actual"); ax[1].set_title("one point per submission"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rounds", type=int, default=14); a = ap.parse_args()
    rdir = HERE / "results"; rdir.mkdir(exist_ok=True); per = []
    for s in a.seeds:
        rep, hist = run(s, a.rounds)
        if s == a.seeds[0]:
            chart(rep, hist, rdir / "simlab_forecast_vs_actual.png")
        sm = rep["summary"]; per.append(dict(**{k: sm[k] for k in ("count", "forecast_mae", "parent_mae", "forecast_wins_vs_parent",
                                                                         "direction_accuracy")}, **rep["toy"]))
        print(f"seed {s}: {per[-1]}")
    summ = dict(seeds=a.seeds, forecast_mae_mean=float(np.mean([p["forecast_mae"] for p in per])),
                parent_mae_mean=float(np.mean([p["parent_mae"] for p in per])),
                direction_accuracy_mean=float(np.mean([p["direction_accuracy"] for p in per])),
                wins=f"{sum(p['forecast_wins_vs_parent'] for p in per)}/{sum(p['count'] for p in per)}", per_seed=per)
    (rdir / "simlab.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "per_seed"}))


if __name__ == "__main__":
    main()

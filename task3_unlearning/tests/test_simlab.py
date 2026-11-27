import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simlab import refit  # noqa: E402
from simlab.score import World  # noqa: E402


def test_score_structure():
    w = World(0); c = w.components(np.array([2.0, 1.5, 1.3, 0.8]))
    assert abs(c["score"] - c["forget"] * c["retain"]) < 1e-12
    assert abs(c["forget"] - c["moments"] * c["prediction"]) < 1e-12
    assert 0 < c["score"] < 1 and 0.28 < w.public.mean() < 0.32


def test_public_split_is_fixed():
    assert np.array_equal(World(1).public, World(1).public)


def test_slope_and_forecast():
    assert abs(refit.slope([0.1, 0.2, 0.3], [0.2, 0.4, 0.6]) - 2.0) < 1e-9
    assert refit.slope([0.1, 0.2], [0.5, 0.5]) == 0.3                    # bounded below
    assert refit.forecast(0.5, 0.4, 0.45, 1.0) == 0.55


def test_backtest_reads_synthetic_receipts(tmp_path):
    rep, hist = refit.run(seed=0, rounds=4, n_init=4, out=tmp_path)
    assert rep["summary"]["count"] == 4 and not rep["issues"]
    for row in rep["rows"]:
        assert row["receipt_id"] > row["basis_receipt_id"]               # every forecast predates its receipt
    jobs = json.loads((tmp_path / "iterations_recus.json").read_text()); assert len(jobs) == 4


def test_backtest_excludes_late_forecasts(tmp_path):
    import backtest_recorded_forecasts as bt
    (tmp_path / "receipt-10.json").write_text(json.dumps(dict(status="success", model="codex_prob_recu_12.pt", submission_id=10, score=0.4)))
    (tmp_path / "iterations_recus.json").write_text(json.dumps({"12": dict(forecast=0.41, parent_score=0.39)}))
    rep = bt.build_report(tmp_path)
    assert rep["summary"]["count"] == 0 and rep["issues"]

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simlab import stats  # noqa: E402
from simlab.toy import T0, Base, build_bench, collect, family_endpoints  # noqa: E402

mp = stats.mp


def test_lcp_fisher_mcc():
    assert mp.lcp("abcd", "abxd") == 2 and mp.lcp([1, 2, 3], [1, 2, 3]) == 3
    assert mp.fisher_z([]) == 0.0 and mp.fisher_z([1e-6, 1e-6]) > 4
    truth = {"a": 1, "b": 1, "c": 2}
    assert mp.mcc(truth, truth)[0] == 1.0
    assert mp.mcc({"a": 1, "b": 2, "c": 1}, truth)[0] < 0


def test_score_truth(tmp_path):
    p = tmp_path / "t.csv"; p.write_text("endpoint,family,model\nx,F,base\ny,F,base\nz,F,lora\nw,G,base\n")
    part = {"x": dict(family="F", model="base"), "y": dict(family="F", model="base"), "z": dict(family="F", model="t"),
            "w": dict(family="G", model="base")}
    assert mp.score_truth(part, p)["score"] == 1.0


def test_bench_is_seeded_and_shaped():
    a, _ = build_bench(3); b, _ = build_bench(3)
    assert len(a) == 33 and [e.role for e in a] == [e.role for e in b]
    assert collect(a[0]) == collect(b[0])
    assert len(collect(a[0])) <= 100


def test_z_is_standard_for_the_base_and_low_when_hot():
    base = Base.random(0); eps = family_endpoints("F", base, 1)
    plain = next(e for e in eps if e.role == "plain"); hot = next(e for e in eps if e.role == "hot")
    P, X, _ = stats.pairs(collect(plain), ("long",)); assert abs(stats.z_at(base.W, base.b_default, T0, P, X)) < 4
    P, X, _ = stats.pairs(collect(hot), ("long",)); assert stats.z_at(base.W, base.b_default, T0, P, X) < -4
    assert abs(stats.calibrate(base, P, X)["T"] - 1.0) < 0.06


def test_fp8_and_watermark_tests():
    base = Base.random(0); eps = family_endpoints("F", base, 1)
    by = {e.role: e for e in eps}
    P, X, _ = stats.pairs(collect(by["fp8"]), ("long",)); assert stats.fp8_test(base, T0, P, X)["z0"] > 4
    P, X, _ = stats.pairs(collect(by["plain"]), ("long",)); assert abs(stats.fp8_test(base, T0, P, X)["z0"]) < 4
    assert stats.repeatability(base, T0, collect(by["watermark"]), 0)["fz"] >= mp.THR["wm_fz"]


def test_family_identification():
    eps, bases = build_bench(0)
    sub = eps[:6]; ans = {e.name: collect(e) for e in sub}
    fam, _, _ = stats.measure(sub, ans, bases)
    assert all(fam[e.name] == e.family for e in sub)

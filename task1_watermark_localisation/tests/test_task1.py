import numpy as np
import pytest

from src.eval_cv import evaluate
from src.metric import miou_batch, miou_image, score
from experiments.visible.src.synth_visible import make_item, make_set, to_cells


def test_metric_perfect_inverted_empty():
    g = np.zeros((8, 8), bool); g[2:5, 2:6] = True
    assert miou_image(g, g) == 1.0
    assert miou_image(~g, g) == 0.0
    assert miou_image(np.zeros_like(g), np.zeros_like(g)) == 1.0
    assert np.allclose(miou_batch(np.stack([g, ~g]), np.stack([g, g])), [1, 0])
    assert score([g], [g]) == 1.0


def test_generator_deterministic_and_exact():
    a = make_item(3, 7); b = make_item(3, 7); c = make_item(3, 8)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    assert not np.array_equal(a[0], c[0])
    x, m, meta = a
    assert x.shape == (256, 256, 3) and x.dtype == np.uint8 and m.shape == (128, 128)
    assert set(np.unique(m)) <= {0, 1}


def test_to_cells_majority():
    m = np.zeros((4, 4), bool); m[:2, :2] = True; m[2, 2] = True
    assert to_cells(m, 2).tolist() == [[1, 0], [0, 0]]


def test_nested_cv_on_synthetic():
    X, M, meta = make_set(30, 5, S=64, cells=32)
    kind = np.array([m["kind"] for m in meta]); fold = np.array([m["fold"] for m in meta])
    kw = dict(G=M, F=fold, me=kind, rg=kind == "tiled", kinds=("text", "logo", "tiled", "clean"), verbose=False)
    res, s = evaluate(M.astype(np.float32), **kw)
    assert res["all"][0] == pytest.approx(1.0)
    has = M.reshape(len(M), -1).any(1) & ~M.reshape(len(M), -1).all(1)
    res_inv, s_inv = evaluate(1 - M.astype(np.float32), **kw)
    assert np.allclose(s_inv[has], 0.0)


def test_predict_shapes_random_init():
    torch = pytest.importorskip("torch")
    from experiments.visible.predict import overlay, predict
    from experiments.visible.src.train_synth import CFG, build_net
    net = build_net(CFG, weights=None).eval()
    card = dict(config=CFG, threshold=0.5)
    img = np.random.default_rng(0).integers(0, 255, (100, 140, 3), dtype=np.uint8)
    p, m = predict(net, card, img)
    assert p.shape == (128, 128) and m.shape == (100, 140)
    assert overlay(img, m).shape == img.shape

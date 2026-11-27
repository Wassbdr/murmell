import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("onnxruntime")

import torch

from experiments.visible.src.audit_watermarks import label_stats
from experiments.visible.src.bench import calibrate, composited_jitter, foreground_iou, measure, measure_onnx, tensors, train_loss
from experiments.visible.src.hard_watermarks import HARD, HELD_OUT, make_scene
from experiments.visible.models import ContrastUnet, build_reference


def test_generator_repeats_and_held_out_has_disjoint_shift():
    assert set(HARD.fonts).isdisjoint(HELD_OUT.fonts)
    assert set(HARD.photos).isdisjoint(HELD_OUT.photos)
    assert HARD.opacity[0] >= HELD_OUT.opacity[1]
    assert HARD.staggered != HELD_OUT.staggered
    for config in (HARD, HELD_OUT):
        scenes = [make_scene(3, i, 64, config) for i in range(24)]
        assert {s.photo for s in scenes if s.textured} == set(config.photos)
        assert {s.kind for s in scenes} == {"text", "logo", "tiled", "clean"}
        for index, scene in enumerate(scenes):
            repeat = make_scene(3, index, 64, config)
            np.testing.assert_array_equal(scene.image, repeat.image)
            np.testing.assert_array_equal(scene.mask, repeat.mask)
            assert scene.image.shape == (64, 64, 3)
            if scene.kind == "clean":
                assert not scene.mask.any()
            else:
                assert config.opacity[0] <= scene.opacity < config.opacity[1]
                assert scene.mask.any()
                assert (scene.alpha[scene.mask] > 0).all()


def test_calibration_and_slice_scoring_use_the_same_threshold():
    scenes = [make_scene(5, i, 64) for i in range(24)]
    masks = np.stack([s.mask for s in scenes])
    probs = np.where(masks, 0.45, 0.25)
    threshold = calibrate(probs, masks)
    assert 0.25 < threshold <= 0.45
    assert measure(probs, scenes, threshold)["all"] == pytest.approx(1)


def test_foreground_iou_has_no_empty_floor():
    masks = np.array([[[0, 0]], [[1, 1]], [[1, 0]]], dtype=bool)
    prediction = np.array([[[0, 0]], [[1, 0]], [[1, 1]]], dtype=bool)
    assert foreground_iou(np.zeros_like(masks), masks) == 0
    assert foreground_iou(prediction, masks) == pytest.approx(0.5)
    assert foreground_iou(prediction[:1], masks[:1]) is None


def test_foreground_iou_is_reported_per_slice():
    scenes = [make_scene(5, i, 64) for i in range(24)]
    masks = np.stack([s.mask for s in scenes])
    scores = measure(masks, scenes, 0.5)
    assert scores["thin text_foreground"] == 1
    assert scores["textured_foreground"] == 1
    assert scores["clean_foreground"] is None


def test_audit_signal_is_measured_at_labelled_pixels():
    scenes = [make_scene(3, i, 64) for i in range(24)]
    marked, nonempty, area, p10_area, signal, p10_signal = label_stats(scenes)
    assert 0 < marked == nonempty < len(scenes)
    assert 0 < p10_area <= area < 1
    assert 0 < p10_signal <= signal <= 255


def test_texture_jitter_preserves_image_range_and_targets():
    image, mask, background, alpha, ink = tensors([make_scene(5, i, 64) for i in range(4)])
    original = alpha.clone()
    augmented, delta = composited_jitter(image.float() / 255, background, alpha, ink, measure_residual=True)
    assert augmented.shape == image.shape
    assert 0 <= augmented.min() <= augmented.max() <= 1
    assert not torch.equal(augmented, image.float() / 255)
    assert torch.equal(alpha, original)
    assert (alpha[mask] > 0).all()
    assert delta is not None
    assert not delta.masked_select(alpha == 0).any()


@pytest.mark.parametrize("name", ["unet", "round1", "slbr", "wdnet", "split"])
def test_cpu_reference_forward_and_backward(name):
    batch = tensors([make_scene(123, i, 64) for i in range(2)])
    model = build_reference(name, 64)
    loss = train_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.eval()(batch[0].float() / 255).shape == batch[1].shape


@pytest.mark.parametrize("residual", [False, True])
def test_contrast_training_and_onnx_probabilities(tmp_path, residual):
    pytest.importorskip("onnx")
    torch.manual_seed(1)
    scenes = {
        "hard": [make_scene(123, i, 64) for i in range(2)],
        "held-out": [make_scene(124, i, 64, HELD_OUT) for i in range(2)],
    }
    model = ContrastUnet(residual=residual)
    train_loss(model, tensors(scenes["hard"])).backward()
    row = measure_onnx(model, tmp_path / "mask.onnx", 0.5, scenes, 1)
    assert row["max_abs_error"] < 1e-5
    assert row["mask_disagreements"] == 0

import sys

import numpy as np
import pytest
import torch
from PIL import Image
from src.invisible_bench import fit_threshold, summarise
from src.invisible_data import HELD_OUT, MAIN, REGIMES, cell_mask, make_invisible_scene
from src.invisible_meta_bench import PixelSealUnet
from src.invisible_product import fit_gate
from src.invisible_references import sliding_scores
from src.metric import score

from murmell.cli import main
from murmell.invisible_model import ResNetUnet
from murmell.localize import search_box
from murmell.onnx import ProbabilityMask, export_mask_model
from murmell.provenance import Family, Gates, Localisation, ProvenanceDetector


class ConstantEmbedder:
    family = "fixture"
    bits = 100

    def embed(self, image, message):
        return Image.new("RGB", image.size, "red")


def test_seeded_scenes_use_disjoint_sources_and_exact_masks():
    assert not set(MAIN.photos) & set(HELD_OUT.photos)
    for config in (MAIN, HELD_OUT):
        for index, regime in enumerate(REGIMES):
            scene = make_invisible_scene(ConstantEmbedder(), 7, index, config)
            repeat = make_invisible_scene(ConstantEmbedder(), 7, index, config)
            assert scene.regime == regime
            assert scene.photo in config.photos
            assert scene.family == "fixture"
            assert scene.donor == (next(p for p in config.photos if p != scene.photo) if regime == "object" else None)
            assert np.array_equal(scene.message, repeat.message)
            assert np.array_equal(scene.image, repeat.image)
            assert 0 < scene.mask.mean() < 1
            assert np.array_equal(scene.mask, cell_mask(scene.support))
            assert score([scene.mask], [repeat.mask]) == 1


def test_cell_coverage_includes_exact_ties():
    support = np.zeros((512, 512), dtype=bool)
    support[:2, :4] = True
    support[4:6, :4] = True
    support[4, 0] = False
    mask = cell_mask(support)
    assert mask.shape == (128, 128)
    assert mask[0, 0] and mask.sum() == 1


def test_sliding_covers_edges_and_scores_the_known_message():
    class Decoder:
        def logits(self, images):
            return torch.tensor([[-1, 1, -1, -1]]).expand(len(images), -1)

    scores = sliding_scores(Decoder(), Image.new("RGB", (512, 256)),
                            np.array([0, 1, 0, 1], dtype=np.uint8), windows=(32,), stride=32)
    assert scores.shape == (128, 128)
    assert np.all(scores == 0.75)


def test_unet_initialises_from_backbone_without_bit_head():
    from torchvision.models import resnet50

    backbone = resnet50(weights=None)
    net = ResNetUnet(backbone).eval()
    assert torch.equal(net.encoder.conv1.weight, backbone.conv1.weight)
    with torch.inference_mode():
        assert net(torch.zeros((1, 3, 32, 32))).shape == (1, 128, 128)


def test_pixelseal_localizer_trains_the_head_and_keeps_the_extractor_frozen():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pwconv2 = torch.nn.Linear(8, 8)

        def forward(self, inputs):
            return self.pwconv2(inputs.movedim(1, -1)).movedim(-1, 1)

    encoder = torch.nn.Module()
    encoder.downsample_layers = torch.nn.ModuleList(
        [torch.nn.Conv2d(3, 8, 4, stride=4)] + [torch.nn.Conv2d(8, 8, 2, stride=2) for _ in range(3)]
    )
    encoder.stages = torch.nn.ModuleList([torch.nn.Sequential(Block()) for _ in range(4)])
    localizer = PixelSealUnet(encoder)
    result = localizer(torch.rand(1, 3, 64, 64))
    assert result.shape == (1, 128, 128)
    result.mean().backward()
    assert localizer.head.weight.grad is not None
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())


def test_benchmark_threshold_uses_competition_scoring():
    target = np.array([[[False, True], [False, True]]])
    prob = np.array([[[0.4, 0.6], [0.4, 0.6]]])
    threshold = fit_threshold(prob, target)
    assert summarise(prob >= threshold, target) == {"miou": 1.0, "foreground_iou": 1.0}
    assert summarise(np.zeros_like(target), target) == {"miou": 0.25, "foreground_iou": 0.0}


def test_box_search_honors_proposal_budget():
    class Decoder:
        count = 0

        def logits(self, images):
            self.count += len(images)
            return torch.tensor([[1, -1]]).expand(len(images), -1)

    decoder = Decoder()
    scores = np.zeros((128, 128), dtype=np.float32)
    scores[30:65, 30:70] = 1
    found = search_box(decoder, Image.new("RGB", (256, 256)), scores, np.array([1, 0], np.uint8), budget=6)
    assert 0 < decoder.count <= 6
    assert found is not None and found.accuracy == 1.0


def test_auto_decode_uses_gate_and_abstains_without_model_inference():
    class Decoder:
        def __init__(self, confidence):
            self.confidence = confidence

        def recover(self, image):
            return np.array([1, 0], np.uint8), self.confidence

        def logits(self, images):
            raise AssertionError("Full-frame decode must not search boxes.")

    class Localizer(torch.nn.Module):
        count = 0

        def forward(self, inputs):
            self.count += 1
            return torch.ones((len(inputs), 128, 128))

    model = Localizer()
    gates = Gates(0.6, 0.5, 0.5, 0.5, 0.8, 1.01)
    detector = ProvenanceDetector({"low": Family(model, Decoder(0.4), gates, 128),
                                   "high": Family(model, Decoder(0.8), gates, 128)})
    image = Image.new("RGB", (32, 24))
    negative = detector.predict(image, "low")
    assert negative.regime == "no confident decode" and negative.mask.getbbox() is None
    assert negative.method is None and model.count == 0
    positive = detector.predict(image)
    assert positive.method == "high" and positive.regime == "full frame with edits"
    assert positive.mask.getbbox() == (0, 0, 128, 128) and model.count == 1

    local = Family(model, Decoder(0.8), gates, 128, lambda _: (8, 0, 24, 24))
    regional = ProvenanceDetector({"high": local}).predict(image)
    assert regional.regime == "local region"
    assert regional.mask.getbbox() == (32, 0, 96, 128) and model.count == 2


def test_product_gate_excludes_clean_calibration_decodes():
    assert fit_gate([0.6, 0.3, -1], [0.3, -1]) == (0.31, 1, 0)


@pytest.mark.parametrize(("confidence", "method", "expected"), [
    (0.4, None, "no confident decode"), (0.8, "trustmark", "trustmark: full frame with edits"),
])
def test_cli_writes_mask_and_reports_decode(monkeypatch, tmp_path, capsys, confidence, method, expected):
    class Stub:
        def predict(self, image, selected):
            assert selected == "auto"
            regime = "full frame with edits" if method else "no confident decode"
            return Localisation(Image.new("L", (128, 128), 255 if method else 0), method, regime, confidence)

    source = tmp_path / "edited.png"
    output = tmp_path / "mask.png"
    Image.new("RGB", (32, 40)).save(source)
    monkeypatch.setattr(ProvenanceDetector, "load", lambda path: Stub())
    monkeypatch.setattr(sys, "argv", ["murmell-detect", str(source), "-o", str(output), "--threads", "1"])
    main()
    assert capsys.readouterr().out.strip() == expected
    with Image.open(output) as mask:
        assert mask.size == (128, 128)
        assert (mask.getbbox() is not None) == (method is not None)


def test_onnx_export_keeps_rgb_normalization(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = torch.nn.Conv2d(3, 1, 1, bias=False)
    torch.nn.init.constant_(model.weight, 1)
    rgb = torch.stack([torch.zeros(3, 128, 128), torch.ones(3, 128, 128)])
    path = tmp_path / "mask.onnx"
    export_mask_model(model, path, 128)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    actual = session.run(["probability"], {"rgb": rgb.numpy()})[0]
    with torch.inference_mode():
        np.testing.assert_allclose(actual, ProbabilityMask(model)(rgb).numpy(), atol=1e-6)

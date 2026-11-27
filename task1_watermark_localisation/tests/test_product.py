import sys

import numpy as np
import pytest
from PIL import Image

from murmell import Detector
from murmell.cli import main


@pytest.mark.parametrize(("variant", "threshold"), [("round1", 0.900), ("contrast", 0.825)])
def test_bundled_detector_returns_input_size_binary_mask(variant, threshold):
    detector = Detector.load(variant)
    assert detector.variant == variant
    assert detector.threshold == threshold
    image = Image.new("RGBA", (45, 31), "#888888")
    mask = detector.predict(image)
    assert mask.size == image.size
    assert mask.mode == "L"
    assert set(np.unique(mask)) <= {0, 255}


@pytest.mark.parametrize(
    ("case", "message"),
    [("missing", "Image file not found"), ("bad", "Cannot read image"), ("weights", "make weights VARIANT=round1")],
)
def test_cli_input_errors(tmp_path, monkeypatch, capsys, case, message):
    image = tmp_path / "image.png"
    if case == "bad":
        image.write_text("not an image")
    if case == "weights":
        Image.new("RGB", (32, 32)).save(image)
    monkeypatch.setattr(sys, "argv", ["murmell-detect", str(image), "-o", str(tmp_path / "mask.png"),
                                    "--weights", str(tmp_path / "missing.pt"), "--threads", "1"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err

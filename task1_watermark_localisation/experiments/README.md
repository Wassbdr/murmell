# Task 1: experiments not in the final pipeline

None of these scripts is used by B28. Results are those of [REPORT.md](../REPORT.md). Run from the task folder:
`python -m experiments.<name>`, or `python experiments/<name>.py` for `snap_full` and `tiny_route`.

| Script | What was tested | Result |
|---|---|---|
| [pseudo_ft.py](pseudo_ft.py) | Self-training on pseudo-labelled test images (PL_* models) | **Tried, excluded by decision, in no submission** (REPORT section 8) |
| [edit_detector.py](edit_detector.py) | Edited-zone detector trained fold by fold on the real masks | AUC 0.49 on missed holes, not used (section 6) |
| [grad_evidence.py](grad_evidence.py) | Decoder bit-gradient evidence maps, PixelSeal and TrustMark | Did not work (section 6, "decoder gradient and occlusion maps") |
| [ps_informed.py](ps_informed.py) | PixelSeal: re-embed the known message at a fitted geometry and correlate with the image | Did not work (section 6) |
| [tm_localize.py](tm_localize.py) | TrustMark 0.9 `localize()` boxes as extra box proposals | No result in REPORT.md |
| [stack.py](stack.py), [stack2.py](stack2.py) | 128×128 stacker on out-of-fold maps; `stack2` is the source behind `src/full_nested.py --stack 1` | Did not work (section 6) |
| [snap_full.py](snap_full.py) | SAM segment or ellipse snapping of the holes in full-regime images | Did not work (section 6: SAM on full-regime holes, ellipse fitting) |
| [tiny_route.py](tiny_route.py) | Tiny-hole specialist map, routed to full-regime images | Did not work (section 6, "a fine-tune specialised on tiny holes") |
| [geo_check.py](geo_check.py), [geo_check2.py](geo_check2.py), [geo_fit.py](geo_fit.py) | Checks of the hypothesis that the real frames are crops of larger watermarked images, resized | No result in REPORT.md; the real processing stays unknown (section 3, step 4) |

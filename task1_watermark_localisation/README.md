# Task 1: watermark localisation

Each test image carries a PixelSeal, TrustMark or StegaStamp watermark over part of the frame. We predict a 128×128 mask
of the watermarked cells. The method and the embedded message are given for every image.

Submission: [submission/B28_final.pt](submission/B28_final.pt). Nested 5-fold CV 0.8285 on the 510 training images,
public leaderboard 0.8127. The CV is optimistic: the final list of chains was shortened after seeing their scores. We
expect about 0.81 on the hidden split.

## Method

1. **Regime.** Decode the whole image and compare with the known message. If it decodes, the whole image is watermarked
   except some holes. If not, only a small region is.
2. **Training data.** We watermark images ourselves with the official embedder and a random message, then cut holes or
   paste regions, so every mask is exact.
3. **Decoder as encoder.** For PixelSeal and TrustMark, the official decoder without its bit head initialises the encoder
   of a U-Net. On TrustMark this alone moves one model from 0.61 to 0.67. StegaStamp uses an ImageNet ResNet-34.
4. **Real images.** Crop-and-resize augmentation (PixelSeal 0.71 to 0.75), soft hole borders (TrustMark), short per-fold
   fine-tuning on the training images, shift test-time augmentation.
5. **Selection.** For each method and regime, models, threshold and post-processing are chosen on 4 folds and scored on
   the fifth. A LightGBM model then picks a mask per image.
6. **Boxes.** In the small regime, a crop that decodes at least 80 % of the message bits becomes the mask.
7. **SAM.** A pasted object carries only a fragment of the watermark and does not decode. We snap the mask to the
   Segment Anything segment that overlaps it best, when they agree. CV 0.803 to 0.819 for this step alone.
   We also train on synthetic pasted objects whose outline is a SAM segment of a real image (`rescale_ft.py --sam_items`):
   TrustMark small images go from 0.847 to 0.855.
8. **Final blend.** Per method and regime, the mask comes from one of a few complete chains, chosen in inner folds. For
   PixelSeal full images, the mean of 27 models beats any selected pair (0.790 against 0.777).

Details, ablations and limits: [REPORT.md](REPORT.md).

## Code

| Step | Files |
|---|---|
| Metric, data, folds | [metric.py](src/metric.py), [data.py](src/data.py), [make_folds.py](src/make_folds.py) |
| Regime | [global_decode.py](src/global_decode.py), [regime_b0.py](src/regime_b0.py) |
| Synthetic data | [synth_full.py](src/synth_full.py), [synth_small.py](src/synth_small.py) |
| U-Net, decoder initialisation, training | [train_full.py](src/train_full.py) |
| Augmentation, fine-tuning, TTA | [rescale_ft.py](src/rescale_ft.py), [finetune_real.py](src/finetune_real.py), [tta.py](src/tta.py) |
| Boxes | [box_search.py](src/box_search.py), [apply_boxes.py](src/apply_boxes.py) |
| SAM | [sam_segments.py](src/sam_segments.py), [sam_ens.py](src/sam_ens.py), [sam_rank.py](src/sam_rank.py), [final_stage/snap_stage.py](src/final_stage/snap_stage.py) |
| Nested selection | [full_nested.py](src/full_nested.py), [nested_cv.py](src/nested_cv.py), [night.py](src/night.py), [selector3.py](src/selector3.py) |
| Final blend (B22) | [final_stage/chains2.py](src/final_stage/chains2.py), [final_stage/snap_stage.py](src/final_stage/snap_stage.py) |
| Submission | [submit.py](src/submit.py) |

The code expects the organisers' starter kit in `kit/`, the data in `release/` and the watermark models in `models/`.

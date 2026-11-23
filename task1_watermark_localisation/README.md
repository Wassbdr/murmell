# Task 1: watermark localisation

Each test image carries a PixelSeal, TrustMark or StegaStamp watermark over part of the frame. We predict a 128×128 mask
of the watermarked cells. The method and the embedded message are given for every image.

Submission: [B28_final.pt](https://github.com/Wassbdr/murmell/releases/tag/task1-best-model), attached to the release `task1-best-model` (16.7 MB; MD5
`bfbf5f3a24b65f406236c83552c82434`, as in [submission/B28_spec.json](submission/B28_spec.json)). Nested 5-fold CV 0.8285 on the 510 training images,
public leaderboard 0.8127. The CV is optimistic: it is nested per stage, not end to end, and the final list of chains was
shortened after seeing their scores. Before the results we expected about 0.81 on the hidden split. The final score on
the hidden 70 % is 0.8328, 2nd of 44 teams on this task (best 0.8524); the ranking does not say which of our submissions
produced it.

## Method

1. **Regime.** Decode the whole image and compare with the known message. If it decodes, the whole image is watermarked
   except some holes. If not, only a small region is.
2. **Training data.** We watermark images ourselves with the official embedder and a random message, then cut holes or
   paste regions, so every mask is exact. The backgrounds are train and test images of the other two methods, used
   without labels.
3. **Decoder as encoder.** For PixelSeal and TrustMark, the official decoder without its bit head initialises the encoder
   of a U-Net. On TrustMark this alone moves one model from 0.61 to 0.67. StegaStamp uses an ImageNet ResNet-34.
4. **Real images.** Crop-and-resize augmentation (PixelSeal 0.71 to 0.75; how the real images were processed is unknown,
   this is a substitute that works), soft hole borders (TrustMark), short per-fold fine-tuning on the training images,
   shift test-time augmentation.
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
| Metric, data, folds | [metric.py](src/metric.py), [data.py](src/data.py), [make_folds.py](src/make_folds.py), [eval_cv.py](src/eval_cv.py) |
| Regime | [global_decode.py](src/global_decode.py), [regime_b0.py](src/regime_b0.py) |
| Synthetic data | [synth_full.py](src/synth_full.py), [synth_small.py](src/synth_small.py), [synth_small2.py](src/synth_small2.py) |
| U-Net, decoder initialisation, training | [train_full.py](src/train_full.py), [assemble_xf.py](src/assemble_xf.py) (assembles the per-fold models) |
| Augmentation, fine-tuning, TTA | [rescale_ft.py](src/rescale_ft.py), [finetune_real.py](src/finetune_real.py), [tta.py](src/tta.py) |
| Combination, clean-up | [combine.py](src/combine.py), [combine2.py](src/combine2.py), [postproc_cv.py](src/postproc_cv.py) (post-processing chosen in inner folds) |
| Boxes | [box_search.py](src/box_search.py), [tm_boxes.py](src/tm_boxes.py), [apply_boxes.py](src/apply_boxes.py) |
| SAM | [sam_segments.py](src/sam_segments.py), [sam_prompt.py](src/sam_prompt.py) (SAM prompted by our maps), [sam_refine.py](src/sam_refine.py), [sam_ens.py](src/sam_ens.py), [sam_rank.py](src/sam_rank.py), [final_stage/snap_stage.py](src/final_stage/snap_stage.py) |
| Nested selection | [full_nested.py](src/full_nested.py), [nested_cv.py](src/nested_cv.py), [night.py](src/night.py), [selector3.py](src/selector3.py) |
| Final blend (B28) | [final_stage/chains2.py](src/final_stage/chains2.py) `--step build`, then [final_stage/snap_stage.py](src/final_stage/snap_stage.py) on each chain, then `chains2.py --step blend`; see [B28_spec.json](submission/B28_spec.json) |
| Submission | [submit.py](src/submit.py); upload tools in [tools/](tools/) |

Other scripts kept in `src/`, not in the B28 chain:

| Script | Role |
|---|---|
| [final_stage/blend_small.py](src/final_stage/blend_small.py), [final_stage/blend_apply.py](src/final_stage/blend_apply.py) | Used for earlier references (B10/B11), not for B28 |
| [final_stage/final_chain.py](src/final_stage/final_chain.py) | Used for an intermediate candidate (B15), not for B28 |
| [selector.py](src/selector.py) | v1 of the per-image selector, used before [selector3.py](src/selector3.py) |
| [final_stage/make_oof.py](src/final_stage/make_oof.py), [final_stage/asm_sel.py](src/final_stage/asm_sel.py), [final_stage/small_pool.py](src/final_stage/small_pool.py) | Role not documented |
| [loss_analysis.py](src/loss_analysis.py) | Breaks down where a candidate loses its points; `make_oof.py` imports its `rebuild` |

Abandoned experiments, including the self-training on test images (tried, excluded by decision, in no submission), are
in [experiments/](experiments/) with one line each on what was tested and the result.

The code expects the organisers' starter kit in `kit/`, the data in `release/` and the watermark models in `models/`.

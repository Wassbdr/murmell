# Task 1 — Can a watermark be localised after the fact?

Team meta-hackathon, CISPA European AI Championship, Paris, 21–22 Nov 2026. Numbers are fully nested 5-fold CV on the 510
train images unless stated; "public" is the public leaderboard (30 % of the test set).

## 1. Why localise

A watermark says "this image comes from generator X". If someone edits a watermarked image, or pastes a watermarked object
into another image, a global detector still fires, and the edit is attributed to the watermark owner, who can be wrongly
accused of content they never produced. Localisation answers the question a detector cannot: *which pixels* carry the mark.

Watermark Anything (WAM) solves this by design: embedder and extractor are trained together so that the extractor outputs a
per-pixel map. The three watermarks of this task (PixelSeal, TrustMark, StegaStamp) were never designed for that: one message
per image, decoders that pool over the whole frame. Our question is therefore **post-hoc localisation**: with only the public
embedders and decoders, general-purpose pretrained vision models, and the message we are told to look for, how well can the mark be localised?

## 2. Answer

Yes, to a useful degree, and how well depends on the carrier and on the scale of the region.

| cell (method, predicted regime) | images | nested mIoU, final candidate | our target |
|---|---|---|---|
| PixelSeal, full frame with holes | 90 | 0.777 | 0.75 |
| PixelSeal, small region | 80 | 0.784 | 0.75 |
| TrustMark, full frame with holes | 88 | 0.754 | 0.78 |
| TrustMark, small region | 82 | 0.840 | 0.80 |
| StegaStamp, full frame with holes | 87 | 0.910 | 0.90 |
| StegaStamp, small region | 83 | 0.881 | 0.82 |
| **all** | 510 | **0.8239** (candidate B10) | 0.83 (not reached) |

Reference: B10, nested 0.8239, public 0.8078 (ID 3574, sent 07:59:52). Before the object-contour step: S11, nested 0.8026,
public 0.7957 (ID 3236). Our honest expectation on the hidden test is about 0.81.

## 3. Method

1. **Informed regime test.** Global decoding agreement with the known message separates "whole frame watermarked, with
   holes" from "small watermarked region" (97 % correct out of fold; its errors cost about 0.004).
2. **Synthetic supervision from the public embedders.** We reproduce the organisers' process (embed a sub-rectangle as an
   image at its own size, then outpainting, transplanted object, or removed / modified / added object) and train
   full-resolution U-Nets on it. These models see no real watermarked/mask pair, but their shape library comes from train masks; some are then fine-tuned on
   real train images, out of fold. A plain U-Net on the 510 real images alone stayed at the constant level (0.45).
3. **Decoder-initialised segmenters (PixelSeal and TrustMark; StegaStamp keeps an ImageNet encoder).** The encoder starts
   from the official decoder (TrustMark ResNet-50, PixelSeal ConvNeXt-V2): the features that read the message also say where it is. Without them: −0.042.
4. **Reducing the synthetic-to-real gap.** The real frames went through a global crop and resize; training with that
   resampling moves a PixelSeal model from 0.7105 to 0.7524 on real full-regime images. Feathered hole borders help
   TrustMark (0.7263 → 0.7334). A light per-fold fine-tuning on the real images adds 0.005 to 0.007, shift TTA a little more.
5. **Informed boxes.** A rectangle search ranked by decoding agreement replaces the mask when a box decodes the message
   with bit accuracy ≥ 0.8. Without it: −0.021.
6. **Object contours from SAM (small regime).** A transplanted object is one object: among the automatic masks of the
   public SAM ViT-H, we take the segment that best overlaps the ensemble of our maps, and a small out-of-fold ranker
   decides between that segment and no snapping. On object images the mask goes from 0.74–0.80 to 0.80–0.89 (per method, T01 against S11);
   PixelSeal rectangles lose on that route (0.709 to 0.662).
   It does not help rectangles or holes, so it is routed per image. Gain: 0.7990 → 0.8112 (fixed rule) → 0.8176 (ranker).
   A second, simpler route gives the same result: snap the final mask to the single SAM segment that overlaps it best,
   if their IoU is at least 0.3 (S11 → 0.8190). The two routes fail on different images; choosing the route per method
   in inner folds, for the small-region cells only, gives 0.8239 (B10).
7. **Fully nested selection.** For each (method, regime) cell, the models, their pairs, the post-processing and the
   threshold are chosen on four folds and scored on the fifth; a rule forbids a model to be used on images whose mask
   shapes it has seen. A per-image mask selector (LightGBM on label-free statistics) added 0.003 to 0.006 before SAM.

## 4. Where the points came from

| step | nested CV | public |
|---|---|---|
| synthetic U-Nets + regime routing (C4r) | 0.6795 | 0.6843 |
| + decoder-ranked boxes (C8) | 0.7004 | 0.6940 |
| + fold-clean full models in small cells (C9) | 0.7233 | 0.7185 |
| + decoder-initialised segmenter, PixelSeal (N00) | 0.7487 | 0.7453 |
| + full-resolution decoder init, real fine-tuning (N01) | 0.7698 | 0.7665 |
| + per-image mask selector (S03) | 0.7828 | 0.7823 |
| + crop-and-resize augmentation (S05) | 0.7884 | 0.7845 |
| + pool pruning, more fold-clean models (S10) | 0.7959 | 0.7953 |
| + fold-clean object items, real fine-tunings (S11) | 0.8026 | 0.7957 |
| + SAM contours for transplanted objects, inside the nested selection (T01) | 0.8176 | 0.8038 |
| + per-image selector on top (T02; +0.0028 over T01, not counted as a gain) | 0.8205 | 0.8050 |
| + second snapping route, route chosen per method; decoder boxes shrunk by 2 px (B10) | 0.8239 | 0.8078 |
| B11, a fourth chain (+0.0011 over B10: within noise, not counted, not the reference) | 0.8250 | 0.8088 |

Figures (in `~/wmloc_report/figures/`): `fig1_timeline.png` nested CV and public score of every submission;
`fig2_gap.png` the synthetic-to-real gap and its closing; `fig3_examples.png` median image of each cell (candidate T01);
`fig4_sam.png` three transplanted objects before and after the SAM contour step (out-of-fold masks of S11 and of the B10 chains; median gain among improved images. Changed object images,
better / worse: PixelSeal 28 / 1, TrustMark 26 / 12, StegaStamp 34 / 5).

## 5. How much to trust 0.80 and 0.82

Thresholds and model choices are nested, but the result is still optimistic, by about 0.015 for B10.

- Selection across passes: S11 is the best of about ten successive passes; recipes (crop-resize, feather, object items) were
  kept after reading validation scores on the same 510 images; 20 selector configurations were looked at.
- The gap nested − public grew with the amount of selection: +0.001 for the first submissions, +0.004 at midnight,
  +0.007 for S11, +0.0161 for B10. From S10 to S11 the nested CV rose by 0.0067 and the public score by 0.0004.
- An independent audit of S11 found no rule leak (no near-duplicate images across folds or with the test set, no side
  channel in files or metadata) and puts its honest value at about 0.796 (range 0.792–0.799); the public score is 0.7957.
- The SAM stage adds its own optimism. Variants looked at: for T01, 8 map variants, prompted masks, 3 rules offered to the
  nested selection and a ranker (a second ranker followed); for B10, 3 successive versions of the snap rule, 324 settings
  chosen in inner folds, and 7 ways of combining chains. The stage, the ranker and the blend sit on out-of-fold outputs whose
  inner folds have seen outer-fold labels, and the variants were designed while looking at the outer folds. The 2-px box shrink was first
  fixed on all 510 images; choosing it in inner folds lowered B10 from 0.8243 to 0.8239 (corrected here, same test file). In B10, TrustMark's
  chain changes in fold 4 and the gain over T02 is negative there (-0.0084). From S11 to B10 the nested CV rose by 0.021 and
  the public score by 0.012: about half. We expect about 0.81 on the hidden part rather than 0.8239.
- The pair applied to the test set is the one chosen on all five folds; it is often not the pair any single fold evaluated.
- One number we first quoted too high: the crop-resize PixelSeal model is 0.7524 as saved (0.7541 was its training peak).

## 6. What did not work

Relative thresholds; re-embedding and correlating (PixelSeal); TrustMark's own remover residual; decoder gradient and
occlusion maps; an edited-zone detector trained on the real masks (AUC 0.49 on missed holes); a 128×128 stacker; top-n
means; lower box thresholds; low-pass and resize-inverse tricks; self-training on test images (not used, by decision);
a "never empty" hole fallback; prompted SAM with a fixed rule; SAM on full-regime holes (holes do not follow objects);
a finer rectangle search by message agreement; TruFor as a zero-shot edit detector (AUC 0.45–0.50 on restored holes); a differential probe with the embedder
(PixelSeal small regions, -0.001).

## 7. Limits

- Thin PixelSeal rectangles: below about 20 px on the short side the decoder cannot read the message and the segmenters
  see almost nothing (28 train images at 0.50–0.57).
- Tiny holes in a fully watermarked frame (less than 3 % of the area): missing one costs half of the image's score;
  75 train images average 0.72.
- TrustMark, full regime (0.754) stays below our target.

## 8. Integrity and disclosures

- No original or clean image was searched or used; ground truth was not reconstructed from outside copies.
- **Self-training was tried.** Four PL_* models were trained on pseudo-labelled test images (04:35–04:42), before the team
  decided against it at 05:50. None is in the lineage of any of our 36 submissions (314 runs traced at 08:49). Otherwise test
  images serve as backgrounds for synthetic data and as inputs at inference.
- **Three submissions had a CV leak** (N08 ID 2140, S08 ID 2908, S09 ID 3007): fine-tuned models used on folds whose mask
  shapes they had seen. Their CV is invalid and they were removed as references. They were sent, so the server still counts them.
- **Public leaderboard.** Models and thresholds come from CV. The public score was watched as an alarm, and at 08:11 it was
  cited when B11 replaced B10 as the reference although the CV gain was below our rule; this was reverted at 08:51.
- **External models and compute.** ImageNet encoders, the official decoders, SAM ViT-H (image only); AWS GPUs besides
  JURECA. SAM was cleared inside the team. We have seen no written organiser rule on external models, and no written
  confirmation on external compute.

## 9. Credits

- Teammate: rectangle verification by agreement with the known message (on the 69 images we could reproduce, 0.9137
  against 0.9157 for our own masks), and a SAM3 post-processing of an earlier candidate (public 0.8015), which pointed us to
  the contour-snapping step.

## 10. Reproduce

B10: `src/final_stage/` (scripts; the base masks of S11 and T02 were rebuilt, per-image scores within 0.0013 of ours, test masks identical) with the inputs in `work/hunt/`, see `work/archive/B10/spec.json`.
`python -m src.night --name <NAME>` rebuilds a loop candidate from the finished runs (`work/runs`), the box files
(`work/boxes`) and the SAM segments (`work/sam`); the exact command of each sent file is in `work/archive/<NAME>/spec.json`.
Compute: two JURECA nodes (4 A100 each) and up to 8 AWS GPUs overnight; one model is 8 to 30 GPU-minutes.

# Task 3: unlearning

A ViT-B/16 classifier (100 classes) must forget 3,500 of its training images: behave on them, and on hidden altered
copies of them, like models retrained from scratch without them ("gold"), while staying identical to the original
everywhere else. No labels, no retain data, no gold model. Score = forget × retain, with forget = (loss mean/std match)
× (per-image prediction match, KL at T = 2), each averaged over exact and altered images, and retain = relative accuracy
× 1 / (1 + KL to the original) on a hidden control set. Below 70 % validation accuracy the score is 0.

**Best public score: 0.4806** (submission 4235). Best model: `cnsu_a32_repair100.pt`, attached to the
release [`task3-best-model`](https://github.com/Wassbdr/murmell/releases/tag/task3-best-model) (328 MB, float32 state dict that loads with
`strict=True`; SHA-256 `57377ff2f5837b9c79ca3f7542aef847c6b28e1caa2022735d56a5d024c79217`). The two previous bests,
`codex_marge_isolee.pt` (0.4795) and `kimi_conditional_soup50.pt` (0.4793), are attached too.

## The idea

A model that never saw these images gets about 4 out of 5 wrong, almost always toward a sibling class (boats 29 to 33,
lanterns 5 to 8, armchairs 34 to 38). Unlearning is not breaking the model: it is teaching it the right mistakes,
image by image, then repairing everything else. On 500 forget images, our final model keeps the original answer only
20 % of the time, close to the gold accuracy.

## Method, stage by stage

| Stage | What it does | Public score | Code |
|---|---|---|---|
| 1. Never-seen targets | The original recognises its images pixel by pixel (shifted 56 px: 39 % still right; blurred: 33 %; grey: 29 %), so on strongly altered views it answers like a model that never saw the image. Per image: label = original on the clean image; wrong classes = original logits averaged over 64 altered views; target loss = difficulty rank-mapped onto the gold loss mean and std | | [unlearn.py](unlearn.py) |
| 2. Distil | Student starts from the original; only the last two ViT blocks and the head are trained, on exact and altered images, toward these targets, with a term that matches the loss mean and std. Mosaics of images anchor it to the original elsewhere | 0.16 → 0.287 | [unlearn.py](unlearn.py), [interp.py](interp.py) |
| 3. Refit loop | After each official result, a few global target parameters (loss moments, temperature, class-size and stability weights) move by bounded steps, then a successor is trained and validated | 0.287 → 0.445 | [refit_loop/](refit_loop/) |
| 4. Retention repair | The control set is new images. On 5,800 images generated in the dataset's style our models drifted (same answer as the original only 65 % of the time). The student keeps the unlearned answer on forget images and their altered copies, and copies the original on generated images. No labels: the original is the teacher | +0.008, +0.013, +0.0135 on three models; 0.4728 | [retention_repair/](retention_repair/) |
| 5. Soups and ensembles | Weight averages of repaired models from different runs, and distillation of the mean of several teachers with the same retention term | 0.4728 → 0.4793 | [soups/](soups/) |
| 6. True-class margin | Head-only, closed-form correction (per-class ridge on penultimate features) that moves each image's true-class margin toward the target loss, on exact and altered views, while the wrong-class distribution barely moves (KL 0.0009) and retention on generated images is unchanged. Held-out on 750 forget images | 0.4793 → 0.4795 | [final_margin/](final_margin/) |
| 7. Projected step + head repair | From the 0.4795 model: one distillation step on the last two blocks and the head, projected away from 32 retention directions (step 0.12), then a head-only repair that brings the loss mean and std back to the previous model's. Chosen on 500 development sources and frozen before any official result; on 500 confirmation sources the KL to both forget targets drops (paired 95 % intervals exclude zero) and retention stays level | 0.4795 → 0.4806 | [final_cnsu/](final_cnsu/) |

### Code map

- [unlearn.py](unlearn.py): targets from altered views, rank mapping onto the gold moments, distillation, mosaic anchors, checks. [interp.py](interp.py) interpolates original and unlearned weights. [tab.py](tab.py) tabulates runs.
- [refit_loop/calibrated_gold.py](refit_loop/calibrated_gold.py): builds the per-image targets from the fitted score model (class features, altered-view probabilities).
- [refit_loop/refit_gold_receipts.py](refit_loop/refit_gold_receipts.py): updates the target parameters from official score differences, with bounded local steps.
- [refit_loop/train_public_proxy.py](refit_loop/train_public_proxy.py): distils the calibrated probability targets while fitting the gold loss moments; splits fit and validation by source image.
- [refit_loop/refine_loss_moments.py](refit_loop/refine_loss_moments.py), [refit_loop/calibrate_losses.py](refit_loop/calibrate_losses.py), [refit_loop/cibles_coherentes.py](refit_loop/cibles_coherentes.py) (one distribution compatible with the KL and loss moments), [refit_loop/three_term_unlearning.py](refit_loop/three_term_unlearning.py) (training with matched original references).
- [retention_repair/t3_gen.py](retention_repair/t3_gen.py): generates new images in the dataset's style with SDXL-Turbo (prompts per class concept).
- [retention_repair/t3_retain.py](retention_repair/t3_retain.py): the repair. [retention_repair/t3_consist.py](retention_repair/t3_consist.py) holds `alter` (crops, flips, small rotations, colour, blur, noise). [retention_repair/t3_alt.py](retention_repair/t3_alt.py) measures behaviour on altered copies. [retention_repair/appliquer_reparation.py](retention_repair/appliquer_reparation.py) applies the repair as a weight delta in 10 seconds, without a GPU.
- [soups/t3_ensemble_retain.py](soups/t3_ensemble_retain.py): distils the mean of several teacher distributions with retention on generated images. [soups/t3_conditional_distill.py](soups/t3_conditional_distill.py): the conditional variant behind the best model. [soups/soup.py](soups/soup.py): weight averaging (the best model is a 50/50 soup). [soups/kimi_opt_head.py](soups/kimi_opt_head.py): regularised head calibration, tested and not kept (it overfit our simulator).
- [final_margin/marge_isolee_finale.py](final_margin/marge_isolee_finale.py): the final head-only margin correction on top of the previous best (anchor), split 2,200 / 550 / 750 forget images for fit, selection and confirmation.
- [final_cnsu/cnsu_phase_a.py](final_cnsu/cnsu_phase_a.py): the projected distillation step (logit gradients, SGD projected on the null space of 32 retention directions, step shrunk on 128 confident controls). [final_cnsu/repair_moments_after.py](final_cnsu/repair_moments_after.py) and [final_cnsu/repair_doses.py](final_cnsu/repair_doses.py): the head repair of the loss moments and its dose selection. [final_cnsu/verification_holdout.py](final_cnsu/verification_holdout.py): the external check on 500 held-out sources.
- [backtest/](backtest/): see below.

## Validation: why this is not tuned on the public 30 %

- **Global parameters only.** The public/private split is done image by image with a hash. Our refit loop moves a few
  global parameters applied to all 3,500 images at once, so it cannot learn which images are public.
- **Forecast before seeing.** Before each submission we forecast its score from earlier results only. On 11 submissions:
  mean error 0.004, against 0.011 for "same score as the parent", closer than the parent 11 times out of 11, right
  direction 11 times out of 11 ([backtest/backtest_recorded_forecasts.py](backtest/backtest_recorded_forecasts.py)).
- **One bench for every candidate.** Same inputs and file hashes for all: retention on 202 generated images never used
  in training, KL to the reference on exact and altered images, forget loss moments
  ([backtest/evaluate_latest_candidates.py](backtest/evaluate_latest_candidates.py),
  [backtest/track3_submission_backtest.py](backtest/track3_submission_backtest.py)). A public difference below 0.004 is
  not read as a signal.
- **It already refused a change.** A per-class head calibration rose in our simulator and fell in public (0.4760): rejected.

## What did not work

Weight noise up to 30 % (no effect), task-vector negation and Newton-Fisher steps (barely move the model), external
teachers DINOv2, CLIP, SigLIP (right 89 % of the time, while gold is wrong about 80 %), forcing the same answer on exact
and altered images (loss moments drift), and eight deliberately unusual methods tested in parallel (nearest unseen
neighbours, imposed loss histograms, pruning memory neurons, input degradation, Langevin annealing): all forget too
little, 39 to 80 % of the images still recognised instead of about 20 %.

## Run

```
python unlearn.py train ...                                     # stage 1 and 2, see --help
python retention_repair/t3_gen.py ...                           # new images, once
python retention_repair/t3_retain.py --base MODEL.pt --gen tr1+test:0:650 --heldout test:650:852 --out repaired \
       --steps 1500 --bf 16 --bg 32 --conf 2 --wmse 1.0 --wf 4 --wr 1.5
python soups/soup.py best.pt distilled.pt parent.pt --weights 0.5,0.5
python backtest/backtest_recorded_forecasts.py                  # reads local receipts only
```

Needs `torch`, `timm`, `diffusers` (generation only), the provided `model.pt`, `forget_images.npz` and `task_template.py`.
The submission key is read from the environment and never committed.

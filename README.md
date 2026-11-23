# Murmell

Team Murmell, European AI Championship (CISPA hackathon), Paris, 21-22 November 2026.

**Slides:** [Murmell - Tasks 1 to 3 (6 min).pdf](slides/Murmell%20-%20Tasks%201%20to%203%20%286%20min%29.pdf)

Final ranking: Murmell is 1st of 44 teams overall, with 0.9819. The overall score is the mean of the three tasks after
rescaling each task so that the best team gets 1 and the last 0.

| Task | Public score | Final (hidden 70 %) | Best solution | Method, code, validation |
|---|---|---|---|---|
| 1. Watermark localisation | 0.8127 | 0.8328, 2nd of 44 | [B28_final.pt](task1_watermark_localisation/submission/B28_final.pt) | [task1_watermark_localisation](task1_watermark_localisation/) |
| 2. Model equality | 1.00 | 1.0000, 1st, tied with 3 teams | [best_submission.csv](task2_model_equality/best_submission.csv) | [task2_model_equality](task2_model_equality/) |
| 3. Unlearning | 0.4806 | 0.4749, 1st of 44 | [cnsu_a32_repair100.pt](https://github.com/Wassbdr/murmell/releases/tag/task3-best-model) (release) | [task3_unlearning](task3_unlearning/) |

The final ranking gives one score per task and does not say which submission produced it: the best solution column is
our reference file, not necessarily the file behind the final score.

## One rule for all three tasks

The public leaderboard sees 30 % of the data; the final ranking uses the other 70 %. We made every decision on data
with known truth, never on the public score:

- Task 1: nested cross-validation on the 510 labelled training images.
- Task 2: the practice endpoints with published answers, and model families we simulated ourselves.
- Task 3: a forecast of every submission from earlier results only (mean error 0.004), and generated images held out
  from training.

## The idea of each task

- **Task 1.** Each test image comes with its watermark method and message, so the official decoder can tell where that
  message still is. We train U-Nets on images we watermark ourselves with the official embedder, starting from the
  decoder's own layers, and snap pasted objects to Segment Anything outlines.
- **Task 2.** We identified the three base models (Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B), run them ourselves and score
  every answer. Disguises (system prompt, persona, watermark, temperature, top_k) and traps (fp8, LoRA doses) become
  statistical tests.
- **Task 3.** A model that never saw the forget images is wrong about 4 times out of 5, toward a sibling class. We teach
  the model those mistakes, estimated from its own answers on strongly altered views, then repair it on generated images
  so it stays identical to the original everywhere else.

## In this repository

Each task folder has a README with the method step by step, links to the code of each step, the validation and the
results. Compute: JURECA and our own GPUs. Datasets and leaderboard keys are not included; the submission scripts read
the key from environment variables.

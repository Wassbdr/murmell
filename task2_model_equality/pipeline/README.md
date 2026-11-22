# Model Equality pipeline (one command, no human in the loop)

`me_pipeline.py` takes raw answers and gives the submission CSV, a diagnosis table with every statistic, and a confidence per pair.
No endpoint id is written anywhere in the code: families, disguises, traps and the endpoints used as adapter anchors are all found
from the answers.

## Command (stable)

```
python me_pipeline.py --config CONFIG.json --answers A.jsonl [B.jsonl ...] --out OUTDIR
                      [--cache DIR] [--truth TRUTH.csv] [--no-lora] [--proxy] [--no-cross] [--redecide] [--set name=value ...]
```

- `--config` : JSON, see below. The endpoints are read from the answer files.
- `--answers`: one or more JSONL files, one answer per line, fields `endpoint`, `id`, `prompt`, `response`, `max_tokens`
  (optional `at` / `sent_at` epoch seconds, used only to find the date printed by date-stamped chat templates).
- `--out`    : output directory (created).
- `--cache`  : directory of per-answer measurements, reused across runs (default `OUTDIR/cache`). Do not share one cache between two
  runs that are running at the same time (whole-file writes: the last one wins).
- `--truth`  : optional CSV `endpoint,family,model[,kind]`; prints and stores the family MCC, the model MCC and the challenge score.
- `--no-lora`: skip stage e (gradient fingerprint). Faster, but adapter traps are then not searched.
- `--proxy`  : also train the adapter proxy (slower, reported next to the fingerprint, not used by the decision). `--no-cross` skips its cross tests.
- `--redecide`: reuse `OUTDIR/state.pkl` (measurements of a previous run) and only redo the decision, for example after `--set`.
- `--set`    : threshold overrides, for example `--set gfp_self=5 gfp_pair=5 gfp_rows=long`.

Needs a GPU, `torch`, `transformers`, `peft`, `numpy`. On the team boxes:
`export HF_HOME=$HOME/hf; /opt/pytorch/bin/python me_pipeline.py ...`. GPU memory stays under 12 GB (checkpoints above 1.7 B
parameters are scored in float16 and differentiated in bfloat16).

## Config

```
{
  "candidates": ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "unsloth/Llama-3.2-1B-Instruct"],
  "endpoints": ["ep-01", "ep-02"],                       (optional: default = every endpoint found in the answer files)
  "reset_markers": ["<|im_start|>system", "<|start_header_id|>system"],   (optional, this is the default)
  "probe_prefixes": ["per-"],                            (optional: ids of the extraction prompts, this is the default)
  "thr": {"gfp_self": 5.0}                               (optional threshold overrides)
}
```

## Outputs (in OUTDIR)

- `partition.csv` : `endpoint,family,model`, ready to submit. `groups.txt`: the same as the argument of `../tests/submit.py`.
- `pairs.csv`     : `endpoint_a,endpoint_b,same_model,confidence` for every pair of the same family.
- `diagnosis.md`  : one table per family with every statistic and the reason of each decision, and the fingerprint pair matrix.
- `diagnosis.json`: the same, machine readable (`families.<F>.endpoints.<endpoint>.measures / flags / fingerprint / decision`,
  `partition`, `score` when `--truth` is given).
- `state.pkl`     : measurements, for `--redecide`.

`show.py OUT/diagnosis.json [TRUTH.csv]` prints one line per endpoint. `compare_lead.py OUT/diagnosis.json BELIEF.json OUT.md`
lists the disagreements with a reference belief. `validate.py` runs the full data and random subsamples at the scored budget.

## How the roles of the answers are found

- reset answer : the prompt contains a reset marker (a fake system turn written with the special tokens of the family).
- long answer  : reset answer that is not a repeatability probe. Used for the temperature.
- repeat group : the same prompt sent at least twice to the same endpoint (at least 4 times to count as a repeatability probe).
- plain answer : no reset marker. Used for the prompt disguise test. Ids starting with `per-` are the extraction prompts.

## Stages

| stage | question | statistic | where it comes from |
|---|---|---|---|
| a. family | which base checkpoint wrote this text | log-likelihood per 1000 characters of the 8 longest answers under every candidate (default chat template, default system prompt, best temperature); the endpoint joins the best candidate. Chat templates that print today's date (Llama 3.2) are rendered with the date that fits the answers best, never with the date of the run | base-id worker |
| (hygiene) | can the end of the answer be read | answers whose text re-tokenizes to more tokens than max_tokens, or whose appended end of turn has a base probability below 0.001, are left out of every token-level statistic: the generated tokens were not the canonical tokenization of the text (6 % of the SmolLM2 answers, 0.2 % of the Qwen and Llama answers). Without this the practice replica looks 4 % too hot and every practice endpoint fails the fingerprint | this pipeline |
| b. prompt disguise | is the default system prompt still there | z of the plain short answers at the temperature fitted on the endpoint's own reset-prefixed long answers; likelihood ratio on the extraction prompts (`per-*`) between the default system prompt and two alternatives (a generic one, an empty one); Fisher z of the extraction prompts against 256 simulated base answers (reported) | lead (base_null.py) + this pipeline |
| b. watermark | does an identical request repeat itself | longest common prefix between answers to one prompt, against 2000 draws of simulated base answers of the same prompt: Fisher z over prompts, or one prompt at the Monte Carlo floor and 6 standard deviations out | refs lab (a fixed "8 characters" rule gives a false alarm on ref-01) |
| b. temperature | which temperature | maximum-likelihood temperature on the reset-prefixed long answers, z at the family temperature (family temperature = lower quartile of the fitted temperatures), stop test | calib.py, refs lab |
| b. truncation | top-k / top-p sampling | tokens sampled beyond rank k (10, 20, 40, 50, 100) or beyond nucleus p (0.80 to 0.99) under the base at the fitted temperature, observed against expected | trunc_test.py (lead) |
| c. fp8 | are these the fp8 weights | likelihood ratio between the exact fp8 copy (vLLM weight-only path: e4m3 weights, one scale per merged q,k,v / o / merged gate,up / down tensor) and the base, with its exact mean and variance under both hypotheses: z0 (text from the base), z1 (text from the fp8 copy), fraction of the way. Fallback recipes when no endpoint of the family is fp8 | refs lab recipe, calibrated here |
| d. adapters | does the endpoint deviate from the base weights, and in whose direction | gradient fingerprint (gradfp.py): score test on a random low-rank sketch of every linear weight, at the maximum-likelihood temperature of the answers that are summed, 24 prompt folds. Self test with its sign-flip p. Calibrated pair test p(a given b) (gradcal.py): fold sums of a against the other folds of b, sign-flip p over the folds of a. Dose ratio with a jackknife standard error | lead + simulated bench |
| e. decision | partition | rules below, then one model label per trap unless two adapter traps have the same direction, a dose ratio of about 1 and the same temperature | |

## Decision rules (in this order, per family)

1. Disguises, all of them mean "base model":
   - watermark: Fisher z of the repeatability test >= 4, or one repeated prompt at the Monte Carlo floor and >= 6 sd above the simulated mean;
   - replaced system context: plain answers z <= -5 at the endpoint's own temperature (persona), or z <= -2.5 together with an alternative
     system prompt that beats the default one by >= 3 nats on the extraction prompts (system prompt). Read before the fingerprint because a
     prompt disguise bends the fingerprint as much as an adapter;
   - truncated sampling: a rank or a nucleus with at least 20 expected tokens beyond it and at most 15 % of them observed.
2. fp8: z0 >= 4 and fraction >= 0.5 on the reset-prefixed answers: trap.
3. Fingerprint self test, sign-flip p <= 0.001 (<= 0.01 when the fitted temperature is away from the family), no disguise: adapter trap.
4. Calibrated pair test p <= 0.02 along one of the two strongest adapter endpoints, with a positive slope: weaker dose of that adapter, trap.
   When no endpoint passes rule 3, the two largest self z above 3 serve as directions.
5. Temperature away from the family with a clean fingerprint: base model sampled at another temperature (disguise).
   A hot endpoint is not an adapter by default: the old rule "the endpoints used to train the adapter proxy are adapters" is gone.
6. Everything else: base model, within the composition prior below.

Composition prior (`max_plain`, default 1): the practice set shows one deployment per configuration, so a family has at most one
endpoint that is the base model with no disguise at all. When several endpoints show neither a disguise nor a trap, the most
base-like one stays in the base group (smallest mean square of the statistics that are N(0,1) for a plain base endpoint: stop test,
quantization test, fingerprint self test and every fingerprint pair test) and the others become models of their own, marked `unres`
(a trap that no test of ours can see, for example a quantization recipe we do not have). `--set max_plain=99` turns the prior off.

When no endpoint of a family carries the vLLM fp8 recipe, the same calibrated test is run with fallback recipes (fp8 with one scale per
layer, fp8 with one scale per output channel, int8 per output channel; list in `quant_recipes` of the config).

## Running on a brand new set of endpoints

Budget: 59 answers per endpoint (16 plain short prompts, 8 identical requests, 35 reset-prefixed long prompts of 256 tokens).
Only box1 talks to the challenge API (`../tests/plan_run.py`, which reads the key itself). The pipeline never calls any API.

```
# 1. phase 1 plan: the 16 plain short prompts (extraction, identity, date, format, random draws, story), no knowledge needed
python make_plan.py 1 plan1.jsonl ep-01 ep-02 ... ep-33
python3 plan_run.py plan1.jsonl s0.jsonl 2                      (box1)

# 2. families from the short answers alone (stage a only needs text): fast run without the fingerprint
python me_pipeline.py --config config.json --answers s0.jsonl --out out1 --no-lora

# 3. phase 2 plan: 8 identical requests of the random-digits prompt and 35 long prompts, behind the reset block of each family
python make_plan.py 2 plan2.jsonl out1/diagnosis.json
python3 plan_run.py plan2.jsonl long.jsonl 2                    (box1)

# 4. the whole pipeline on everything
python me_pipeline.py --config config.json --answers s0.jsonl long.jsonl --out out2 --cache cache
#    out2/partition.csv is the submission, out2/diagnosis.md the reasons, out2/pairs.csv the confidence per pair
```

`config.json` only needs the candidate base checkpoints (`config_scored.json` is the one used for the 33 endpoints). If the
candidates are unknown, put every plausible small instruct model in the list: stage a keeps the best one per endpoint and reports
the margin to the runner-up in `diagnosis.json` (`family.margin_per_kchar`).

Why these prompts:
- the plain short prompts see the provider's system context (extraction, identity, date) and are the only place where a prompt
  disguise is fully visible;
- the reset block (a fake end of turn, a fake neutral system turn, a fake user header written with the special tokens of the family)
  removes most of a prompt disguise, so the long answers measure the weights and the sampler;
- 8 identical requests of a high-entropy prompt (random digits) settle the watermark: the same beginning comes back;
- long answers of 256 tokens carry the tokens that the temperature, truncation, fp8 and fingerprint tests need (about 9 000 per endpoint).

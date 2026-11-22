# Task 2: model equality

33 black-box LLM endpoints. Group the endpoints that serve the same model. A model can be disguised (system prompt,
persona, watermark, sampling settings), and different models can be close (fp8 quantisation, LoRA at several strengths).

Public score: 1.00.

## Result

Three families of 11 endpoints: Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct. In each family,
6 endpoints serve the same base model and form one group. The fp8 version and the 4 LoRA strengths are separate models.

## Tests

Each test was checked first on the 9 training endpoints, whose answers are published.

- Likelihood of the answers under each base model: finds the family.
- Hidden-prompt extraction and repeated prompts: repeatability shows the watermark, varying answers show the temperature.
- Truncation test: shows top-k and top-p.
- fp8 versus full-precision likelihood ratio: finds the quantised model.
- Chat-template tokens in the prompt and a gradient fingerprint: find the LoRA and order its strengths.

## Files

- [structure_endpoints.csv](structure_endpoints.csv): family, base model and role of each endpoint.
- [soumissions_tache2.csv](soumissions_tache2.csv): every submission with its score and groups.
- [submitted_files](submitted_files/): the files as submitted.
- `reponses_endpoints_*.csv`: the 2,909 prompts and answers collected.

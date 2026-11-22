#!/usr/bin/env bash
set -euo pipefail

receipt=$1
parent=$2
fraction=$3
retention_weight=$4
anchor_strategy=${5:-uniform}
ensemble_strategy=${6:-auto}
attempt=${7:-initial}
stage=${8:-train}
tag="codex_prob_recu_${receipt}"
comparison_id=$receipt
if test "$attempt" = retry_alt; then
    tag="${tag}_reprise"
    comparison_id="${receipt}_reprise"
fi
cd /workspace/unlearning
prefix=s3://your-bucket/track3-unlearning
dir=tools/codex_gold
eval_dir=tools/codex_compare_lina
python=python
gpu=track3-gpu
export PYTHONPATH="$PWD/tools/codex_three_record:$PWD/tools"
if test "$ensemble_strategy" = auto; then
    ensemble_strategy=$("$python" -c 'import json,sys; from pathlib import Path; items=list(Path("output/receipts").glob("*."+sys.argv[1]+".json")); print("include_parent" if items and not json.loads(items[0].read_text()).get("improved",True) else "current")' "$receipt")
fi

copy_to_s3() {
    aws --region us-east-1 s3 cp "$1" "$prefix/$2" --only-show-errors
}

archive() {
    for file in "output/$tag.log" "subs/$tag.json" "output/codex_fit_recu_$receipt.json" "output/codex_gold_public_$receipt.json"; do
        if test -s "$file"; then copy_to_s3 "$file" "aws-logs/$(basename "$file")"; fi
    done
}
trap archive EXIT

if test "$attempt" = retry && ! test -s "subs/$tag.pt"; then
    for form in "subs/$tag.json" "output/$tag.log"; do
        if test -s "$form"; then
            copy_to_s3 "$form" "aws-logs/${tag}_essai_initial.${form##*.}"
        fi
    done
fi

exec {fit_lock}>"cache/codex_gold_public_${receipt}.lock"
flock "$fit_lock"
for file in refit_gold_receipts.py calibrated_gold.py train_public_proxy.py mesurer_depuis_prefixe.py collect_validation_logits.py; do
    aws --region us-east-1 s3 cp "$prefix/codex/calibrated_gold/$file" "$dir/$file" --only-show-errors
done

if ! test -s "cache/codex_gold_public_$receipt.pt"; then
    scored_model=$("$python" - "$receipt" <<'PY'
import json
from pathlib import Path
import re
import sys
number = int(sys.argv[1])
records = [json.loads(path.read_text()) for path in Path("output/receipts").glob(f"*.{number}.json")]
names = {r["model"] for r in records if r.get("status") == "success" and r["submission_id"] == number}
if len(names) != 1:
    raise ValueError(f"Expected one evaluated model for receipt {number}")
name = names.pop()
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.pt", name):
    raise ValueError(f"Invalid model filename for receipt {number}")
print(name)
PY
)
    private="cache/codex_prevision_privee/cache/lina_feat/$scored_model"
    common="cache/codex_controle_familles/$scored_model"
    baseline="cache/lina_feat/$scored_model"
    if ! test -s "$private" && ! test -s "$common" && ! test -s "$baseline"; then
        spec="$eval_dir/receipt_${receipt}_source.txt"
        printf '%s|subs/%s|1|1|?\n' "${scored_model%.pt}" "$scored_model" > "$spec"
        "$gpu" bash -c '
            set -euo pipefail
            if ! test -s "$1" && ! test -s "$2" && ! test -s "$3"; then
                python -u "$4" "$5"
            fi
        ' -- "$private" "$common" "$baseline" "$eval_dir/collect_current.py" "$spec" \
            > "output/codex_receipt_${receipt}_features.log" 2>&1 &
        feature_worker=$!
        while kill -0 "$feature_worker" 2>/dev/null; do
            if test -s "$private" || test -s "$common" || test -s "$baseline"; then
                if CUDA_VISIBLE_DEVICES="" "$python" - "$private" "$common" "$baseline" <<'PY'
from pathlib import Path
import sys
import torch
for name in sys.argv[1:]:
    p = Path(name)
    if p.is_file():
        try:
            f = torch.load(p, map_location="cpu", weights_only=True)
            if f["clean"].shape == (3500, 100):
                sys.exit(0)
        except (EOFError, KeyError, OSError, RuntimeError):
            continue
sys.exit(1)
PY
                then
                    break
                fi
            fi
            sleep 2
        done
        if ! kill -0 "$feature_worker" 2>/dev/null; then wait "$feature_worker"; fi
    fi
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$python" "$dir/refit_gold_receipts.py" --through "$receipt" --require-receipt > "output/$tag.log" 2>&1
    test -s "cache/codex_fit_recu_$receipt.pt"
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$python" "$dir/calibrated_gold.py" --fit-file "cache/codex_fit_recu_$receipt.pt" --output "cache/codex_gold_public_$receipt.pt" >> "output/$tag.log" 2>&1
    copy_to_s3 "cache/codex_gold_public_$receipt.pt" "caches/codex_gold_public_$receipt.pt"
fi
flock -u "$fit_lock"
exec {fit_lock}>&-
if test "$stage" = prepare_targets; then exit 0; fi
if ! test -s "subs/$tag.pt"; then
    extra_args=(--receipt "$receipt" --share-frozen-blocks)
    epoch_limit=4
    parent_report="subs/${parent%.pt}.json"
    if test -s "$parent_report"; then
        settings=$("$python" - "$parent_report" <<'PY'
import json
import math
from pathlib import Path
import sys
report = Path(sys.argv[1])
parent = json.loads(report.read_text())
seen = {report.name}
while parent.get("training_configuration_source") and parent.get("source"):
    source = Path(parent["source"])
    if source.name != str(source) or source.suffix != ".pt":
        break
    source_report = report.with_name(source.with_suffix(".json").name)
    if source_report.name in seen or not source_report.is_file():
        break
    seen.add(source_report.name)
    parent = json.loads(source_report.read_text())
method = "stratified" if parent.get("stratify_loss") else "population" if parent.get("population_moments") else "ordinary"
epochs = parent.get("epochs", [])
limit = 4
if method != "ordinary" and len(epochs) >= 2:
    selected = parent.get("selected_epoch") or 0
    last = epochs[-1]
    if selected == last["epoch"] and last["quality"] > epochs[-2]["quality"]:
        limit = min(8, max(4, math.ceil(selected) + 2))
    else:
        limit = min(8, max(4, math.ceil(selected) + 1))
temperature = parent.get("decision_temperature")
print(method, limit, temperature if temperature is not None else "none")
PY
)
        read -r inherited epoch_limit decision_temperature <<< "$settings"
        case "$inherited" in
            population) extra_args+=(--population-moments) ;;
            stratified) extra_args+=(--population-moments --stratify-loss) ;;
        esac
        if test "$decision_temperature" != none; then
            extra_args+=(--decision-temperature "$decision_temperature")
        fi
    fi
    if test "$ensemble_strategy" = include_parent; then
        reference=$("$python" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text()).get("gold_cache","cache/codex_gold_public.pt") if p.is_file() else "cache/codex_gold_public.pt")' "subs/${parent%.pt}.json")
        if [[ "$reference" == cache/*.pt && "$reference" != *..* && "$reference" != "cache/codex_gold_public_$receipt.pt" ]] && test -s "$reference"; then
            extra_args+=(--reference-gold "$reference")
        fi
    fi
    "$gpu" python -u "$dir/train_public_proxy.py" "subs/$parent" "subs/$tag.pt" \
        --gold-cache "cache/codex_gold_public_$receipt.pt" --fraction "$fraction" --epochs "$epoch_limit" \
        --retention-weight "$retention_weight" --anchor-strategy "$anchor_strategy" \
        "${extra_args[@]}" >> "output/$tag.log" 2>&1
fi
if ! test -s "subs/$tag.pt"; then
    echo "Aucun modèle n'a dépassé le parent en validation" >> "output/$tag.log"
    exit 0
fi
copy_to_s3 "subs/$tag.pt" "checkpoints/$tag.pt"
sha256sum "subs/$tag.pt" > "output/$tag.sha256"
copy_to_s3 "output/$tag.sha256" "aws-logs/$tag.sha256"
archive
if ! test -s "cache/supplement_$comparison_id.pt"; then
    "$gpu" python -u "$dir/collect_validation_logits.py" "cache/supplement_$comparison_id.pt" "subs/$tag.pt" >> "output/$tag.log" 2>&1
fi
copy_to_s3 "cache/supplement_$comparison_id.pt" "caches/supplement_$comparison_id.pt"

spec="$eval_dir/$tag.txt"
printf '%s|subs/%s.pt|1|1|?\n' "$tag" "$tag" > "$spec"
"$gpu" python -u "$eval_dir/collect_current.py" "$spec" >> "output/$tag.log" 2>&1
aws --region us-east-1 s3 cp "$prefix/codex/calibrated_gold/rank_challengers.py" "$eval_dir/rank_challengers.py" --only-show-errors
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$python" "$eval_dir/rank_challengers.py" "${tag}_classement" "$tag" >> "output/$tag.log" 2>&1
copy_to_s3 "output/${tag}_classement.json" "aws-logs/${tag}_classement.json"
if test -s cache/codex_fit_recu_2081.pt; then
    mapfile -t recent_models < <("$python" -c 'from pathlib import Path; names=[f.stem for f in Path("cache/codex_controle_familles").glob("codex_prob_recu_*.pt") if f.stem.rsplit("_",1)[-1].isdigit()]; print("\n".join(sorted(names,key=lambda n:int(n.rsplit("_",1)[-1]),reverse=True)[:5]))')
    if CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$python" "$dir/refit_gold_receipts.py" --through "$receipt" --base-through 2081 --forecast "$tag" "${recent_models[@]}" --forecast-output "output/${tag}_fige.json" >> "output/$tag.log" 2>&1; then
        copy_to_s3 "output/${tag}_fige.json" "aws-logs/${tag}_fige.json"
    fi
fi

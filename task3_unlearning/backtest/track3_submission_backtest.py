#!/usr/bin/env python3
"""Unified track 3 public receipts, pushed checkpoints, and candidate validation report."""

import argparse
import datetime as dt
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "backtest_inputs"


def load_public_history():
    records = {}
    for path in sorted(ROOT.glob("receipt-*.json")):
        data = json.loads(path.read_text())
        if data.get("status") != "success":
            continue
        receipt_id = int(data["submission_id"])
        records[receipt_id] = {
            "id": receipt_id,
            "name": Path(data["model"]).stem,
            "model": data["model"],
            "parent_model": None,
            "public_score": float(data["score"]),
            "source": path.name,
            "match": "official_receipt_file",
        }
    for path in sorted(INPUTS.glob("receipts/*.json")):
        data = json.loads(path.read_text())
        if data.get("status") != "success":
            continue
        receipt_id = int(data["submission_id"])
        records[receipt_id] = {
            "id": receipt_id,
            "name": Path(data["model"]).stem,
            "model": data["model"],
            "parent_model": None,
            "public_score": float(data["score"]),
            "source": str(path),
            "match": "official_s3_receipt",
        }
    cache_path = INPUTS / "official-receipt-cache.json"
    if cache_path.is_file():
        for data in json.loads(cache_path.read_text()):
            records[int(data["id"])] = data
    likely_path = INPUTS / "likely_receipt_assignments.json"
    likely = json.loads(likely_path.read_text()) if likely_path.is_file() else {}
    sendloop = INPUTS / "box1-sendloop.log"
    if sendloop.is_file():
        for line in sendloop.read_text().splitlines():
            try:
                data = json.loads(line.split(" ", 1)[1])
            except (ValueError, IndexError):
                continue
            if data.get("status") != "success" or "submission_id" not in data:
                continue
            receipt_id = int(data["submission_id"])
            assignment = likely.get(str(receipt_id), {})
            model = assignment.get("model")
            records[receipt_id] = {
                "id": receipt_id,
                "name": Path(model).stem if model else "box1_submission_model_to_confirm",
                "model": model,
                "parent_model": None,
                "public_score": float(data["score"]),
                "source": sendloop.name,
                "match": assignment.get("confidence", "official_score_model_not_present_in_log"),
                "assignment_evidence": assignment.get("evidence"),
            }
    for data in json.loads((INPUTS / "manual_receipts.json").read_text()):
        records[int(data["id"])] = data
    history = [records[key] for key in sorted(records)]
    temporary = cache_path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(history, indent=2) + "\n")
    temporary.replace(cache_path)
    return history


def load_inventory(path):
    artifacts = []
    if not path.is_file():
        return artifacts
    for line in path.read_text().splitlines():
        cells = line.split(None, 3)
        if len(cells) != 4:
            continue
        date, time, size, key = cells
        if not key.endswith(".pt") or not size.isdigit():
            continue
        size = int(size)
        if not 300_000_000 <= size <= 450_000_000:
            continue
        if "claude_delta_reparation" in key or key.endswith("/model.pt"):
            continue
        artifacts.append({
            "key": key,
            "name": Path(key).stem,
            "size": size,
            "last_modified": date + "T" + time + "+00:00",
        })
    artifacts.sort(key=lambda row: row["last_modified"], reverse=True)
    return artifacts


def load_candidate_evaluation():
    paths = sorted(INPUTS.glob("box1/out/*/track3_backtest_*.json"))
    reports = []
    for path in paths:
        try:
            data = json.loads(path.read_text())
            data["report_path"] = str(path)
            reports.append(data)
        except json.JSONDecodeError:
            continue
    if not reports:
        return None
    reports.sort(key=lambda row: row.get("generated_at", ""))
    merged = dict(reports[-1])
    merged["reports"] = [row["report_path"] for row in reports]
    merged["candidates"] = {}
    for report in reports:
        for name, row in report.get("candidates", {}).items():
            row = dict(row, evaluation_report=report["report_path"],
                       baseline_name=report["baseline"]["name"])
            merged["candidates"][name] = row
    return merged


def load_l40s_reports():
    path = ROOT / "updates/l40s-candidate-reports.jsonl"
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        data = row.get("data", {})
        if "student" not in data or "base" not in data:
            continue
        rows.append({
            "name": Path(row["file"]).stem,
            "args": data.get("args"),
            "base": data["base"],
            "student": data["student"],
            "dist": data.get("dist"),
        })
    return rows


def leaderboard_score(task_id):
    path = INPUTS / "leaderboard.html"
    if not path.is_file():
        return None
    html = path.read_text()
    for block in re.findall(r'<tr[^>]*data-team="Murmell"[\s\S]*?</tr>', html):
        if "c-overall" not in block:
            continue
        match = re.search(r'data-task="' + re.escape(task_id) +
                          r'"[^>]*>\s*<span class="tile[^"]*" title="([0-9.]+)"', block)
        if match:
            return {"score": float(match.group(1)), "model": None,
                    "match": "public_leaderboard_model_not_shown"}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INPUTS / "s3-inventory.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "output/track3_submission_backtest.json")
    args = parser.parse_args()
    history = load_public_history()
    artifacts = load_inventory(args.inventory)
    evaluation = load_candidate_evaluation()
    l40s_reports = load_l40s_reports()
    by_model = {}
    for row in history:
        if row.get("model"):
            by_model[Path(row["model"]).stem] = row
    for row in artifacts:
        receipt = by_model.get(row["name"])
        row["public_score"] = receipt["public_score"] if receipt else None
        row["receipt_id"] = receipt["id"] if receipt else None
        row["evaluated_on_box1"] = bool(evaluation and row["name"] in evaluation["candidates"])

    best = max(history, key=lambda row: row["public_score"])
    latest = max(history, key=lambda row: row["id"])
    leaderboard = leaderboard_score("31-unlearning")
    current = best
    if leaderboard and leaderboard["score"] > best["public_score"]:
        current = {
            "id": None,
            "name": "Murmell leaderboard, model to confirm",
            "public_score": leaderboard["score"],
            "source": "public leaderboard",
            "match": leaderboard["match"],
        }
    parent_scores = {row["name"]: row["public_score"] for row in history}
    repair_pairs = []
    for row in history:
        parent = row.get("parent_model")
        if parent and parent in parent_scores:
            repair_pairs.append({
                "parent": parent,
                "repaired_receipt": row["id"],
                "parent_public": parent_scores[parent],
                "repaired_public": row["public_score"],
                "gain": row["public_score"] - parent_scores[parent],
                "evidence": row["match"],
            })
    forecast_path = ROOT / "output/recorded_forecast_backtest.json"
    forecast = json.loads(forecast_path.read_text())["summary"] if forecast_path.is_file() else None
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Public receipts and common-baseline validation. No private labels and no synthetic official score.",
        "coverage": {
            "public_receipts": len(history),
            "full_checkpoint_artifacts": len(artifacts),
            "checkpoints_with_public_receipt": sum(row["receipt_id"] is not None for row in artifacts),
            "checkpoints_evaluated_on_box1": sum(row["evaluated_on_box1"] for row in artifacts),
            "l40s_candidate_reports": len(l40s_reports),
        },
        "current_public_record": current,
        "best_identified_receipt": best,
        "leaderboard_current": leaderboard,
        "latest_public_receipt": latest,
        "repair_evidence": {
            "pairs": repair_pairs,
            "mean_gain": sum(row["gain"] for row in repair_pairs) / len(repair_pairs) if repair_pairs else None,
            "warning": "The D series is already repaired according to NOTE-REPARATION-INSTANTANEE.md; do not reapply blindly",
        },
        "recorded_public_forecast_backtest": forecast,
        "box1_common_baseline_evaluation": evaluation,
        "l40s_candidate_reports": l40s_reports,
        "latest_checkpoint_artifacts": artifacts[:40],
        "public_history": history,
        "limitations": [
            "Public scores are official; local retention and forget metrics are diagnostics, not official scores",
            "The box1 comparison uses a frozen repaired baseline and one held-out generated set",
            "A pushed checkpoint without a receipt or common-baseline evaluation is marked unmeasured, not scored by name",
            "New official submissions whose logs omit the model name remain explicitly unassigned",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.json")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown(report))
    print("Public receipts", len(history), "record", best["public_score"], best["name"])
    print("Checkpoint artifacts", len(artifacts), "box1 evaluations", evaluation and len(evaluation["candidates"]))
    print("Report", args.output)
    print("Summary", markdown_path)


def markdown(report):
    best = report["current_public_record"]
    repair = report["repair_evidence"]
    lines = [
        "# Track 3 - Submissions, nouveaux checkpoints et backtest commun",
        "",
        f"Genere: {report['generated_at']}",
        "",
        f"- Recus publics: {report['coverage']['public_receipts']}",
        f"- Checkpoints complets inventories: {report['coverage']['full_checkpoint_artifacts']}",
        f"- Record public actuel: {best['name']}, recu {best['id']}, score {best['public_score']:.4f}",
        f"- Dernier recu identifie: {report['best_identified_receipt']['name']}, recu {report['best_identified_receipt']['id']}, score {report['best_identified_receipt']['public_score']:.4f}",
        "",
        "## Reparation de retention",
        "",
        "| Parent | Score parent | Score repare | Gain | Recu |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in repair["pairs"]:
        lines.append(f"| {row['parent']} | {row['parent_public']:.4f} | {row['repaired_public']:.4f} | "
                     f"{row['gain']:+.4f} | {row['repaired_receipt']} |")
    lines.extend(["", f"Gain moyen: {repair['mean_gain']:+.6f}. {repair['warning']}", ""])
    evaluation = report["box1_common_baseline_evaluation"]
    if evaluation:
        baseline = evaluation["baseline"]
        lines.extend([
            "## Evaluation commune sur box1",
            "",
            f"Reference: {baseline['name']}, SHA256 `{baseline['sha256']}`.",
            "",
            "| Candidat | Retain estime | Delta retain | KL exact | KL altere | Delta CE exacte | Delta CE alteree |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name, row in evaluation["candidates"].items():
            if row.get("status") != "success":
                continue
            metrics = row["metrics"]
            delta = row["delta_vs_baseline"]
            lines.append(f"| {name} | {metrics['neuves']['retain_estime']:.5f} | {delta['retain_estime']:+.5f} | "
                         f"{metrics['oubli']['kl_base_eleve_exactes']:.5f} | "
                         f"{metrics['oubli']['kl_base_eleve_alterees']:.5f} | "
                         f"{delta['ce_exact_mean']:+.5f} | {delta['ce_altered_mean']:+.5f} |")
        lines.extend(["", "Ces metriques comparent les modeles sur les memes entrees. Elles ne sont pas un score officiel synthetique.", ""])
    lines.extend([
        "## Derniers checkpoints pousses",
        "",
        "| Modele | Date UTC | Public | Evalue box1 |",
        "|---|---|---:|---:|",
    ])
    for row in report["latest_checkpoint_artifacts"][:20]:
        public = f"{row['public_score']:.4f}" if row["public_score"] is not None else "non soumis"
        lines.append(f"| {row['name']} | {row['last_modified']} | {public} | {row['evaluated_on_box1']} |")
    lines.extend([
        "",
        "Les checkpoints sans recu ni evaluation commune restent marques comme non mesures. Aucun score ne leur est attribue a partir de leur nom.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()

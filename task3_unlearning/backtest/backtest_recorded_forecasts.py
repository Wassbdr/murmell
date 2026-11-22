#!/usr/bin/env python3
"""Audit recorded candidate forecasts against later official public receipts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def mean(values):
    return sum(values) / len(values) if values else None


def mae(errors):
    return mean([abs(value) for value in errors])


def rmse(errors):
    return math.sqrt(mean([value * value for value in errors])) if errors else None


def load_receipts(root: Path):
    receipts = {}
    issues = []
    for path in sorted(root.glob("receipt-*.json")):
        data = json.loads(path.read_text())
        if data.get("status") != "success" or not data.get("model"):
            continue
        name = Path(data["model"]).stem
        row = {
            "id": int(data["submission_id"]),
            "model": name,
            "score": float(data["score"]),
            "receipt_file": path.name,
        }
        if not 0 <= row["score"] <= 1:
            raise ValueError(f"Invalid score in {path}")
        previous = receipts.get(name)
        if previous and previous["score"] != row["score"]:
            issues.append(f"Different scores for repeated model {name}; latest receipt used")
        if previous is None or row["id"] > previous["id"]:
            receipts[name] = row
    return receipts, issues


def frozen_fit_checks(root: Path):
    checks = []
    for path in sorted(root.glob("codex_prevision_comparee_*.json")):
        data = json.loads(path.read_text())
        frozen_at = data.get("frozen_at")
        for row in data.get("historical", []):
            if frozen_at is None or int(row["id"]) <= int(frozen_at):
                continue
            error = float(row["median"]) - float(row["actual"])
            checks.append({
                "source": path.name,
                "frozen_at": int(frozen_at),
                "receipt_id": int(row["id"]),
                "actual": float(row["actual"]),
                "forecast": float(row["median"]),
                "error": error,
            })
    errors = [row["error"] for row in checks]
    return {
        "count": len(checks),
        "mae": mae(errors),
        "rmse": rmse(errors),
        "bias": mean(errors),
        "checks": checks,
    }


def build_report(root: Path):
    receipts, issues = load_receipts(root)
    jobs_path = root / "iterations_recus.json"
    jobs = json.loads(jobs_path.read_text())
    rows = []
    for basis, job in sorted(jobs.items(), key=lambda item: int(item[0])):
        if not isinstance(job.get("forecast"), (int, float)):
            continue
        model = f"codex_prob_recu_{basis}"
        if job.get("attempt") == "retry_alt":
            model += "_reprise"
        receipt = receipts.get(model)
        if receipt is None:
            continue
        if receipt["id"] <= int(basis):
            issues.append(f"Excluded {model}: receipt {receipt['id']} is not after basis {basis}")
            continue
        parent_score = float(job["parent_score"])
        forecast = float(job["forecast"])
        actual = receipt["score"]
        forecast_error = forecast - actual
        parent_error = parent_score - actual
        predicted_change = forecast - parent_score
        observed_change = actual - parent_score
        rows.append({
            "basis_receipt_id": int(basis),
            "receipt_id": receipt["id"],
            "model": model,
            "parent": job.get("parent"),
            "parent_score": parent_score,
            "forecast": forecast,
            "actual": actual,
            "predicted_change": predicted_change,
            "observed_change": observed_change,
            "forecast_error": forecast_error,
            "parent_error": parent_error,
            "forecast_beats_parent_absolute_error": abs(forecast_error) < abs(parent_error),
            "direction_correct": ((predicted_change >= 0) == (observed_change >= 0)),
        })
    rows.sort(key=lambda row: row["receipt_id"])
    forecast_errors = [row["forecast_error"] for row in rows]
    parent_errors = [row["parent_error"] for row in rows]
    forecast_mae = mae(forecast_errors)
    parent_mae = mae(parent_errors)
    return {
        "scope": "Recorded forecasts of public scores, not private leaderboard scores",
        "evidence_level": "Forecasts stored in iterations_recus.json and matched to later receipts; no signed timestamps",
        "summary": {
            "count": len(rows),
            "forecast_mae": forecast_mae,
            "forecast_rmse": rmse(forecast_errors),
            "forecast_bias": mean(forecast_errors),
            "forecast_median_absolute_error": statistics.median(map(abs, forecast_errors)) if rows else None,
            "parent_mae": parent_mae,
            "parent_bias": mean(parent_errors),
            "mae_skill_vs_parent": 1 - forecast_mae / parent_mae if forecast_mae is not None and parent_mae else None,
            "forecast_wins_vs_parent": sum(row["forecast_beats_parent_absolute_error"] for row in rows),
            "direction_accuracy": mean([float(row["direction_correct"]) for row in rows]),
        },
        "rows": rows,
        "frozen_fit_backtest": frozen_fit_checks(root),
        "issues": issues,
        "interpretation": [
            "The parent baseline predicts that a candidate keeps its parent's public score",
            "A low public forecast error does not measure public-to-private generalization",
            "Use fresh source-grouped image holdouts and unseen alterations before selecting a final private candidate",
        ],
    }


def markdown(report):
    summary = report["summary"]
    lines = [
        "# Backtest des previsions enregistrees du track 3",
        "",
        "Ce rapport concerne les scores publics. Il ne mesure pas directement le transfert au leaderboard prive.",
        "",
        f"- Previsions appariees avec un recu posterieur: {summary['count']}",
        f"- MAE des previsions: {summary['forecast_mae']}",
        f"- MAE du baseline parent: {summary['parent_mae']}",
        f"- Reduction de MAE par rapport au parent: {summary['mae_skill_vs_parent']:.2%}",
        f"- Biais des previsions: {summary['forecast_bias']}",
        f"- Bonne direction progression/regression: {summary['direction_accuracy']:.2%}",
        f"- Previsions plus proches que le parent: {summary['forecast_wins_vs_parent']}/{summary['count']}",
        "",
        "| Recu | Modele | Parent | Prevision | Officiel | Erreur |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(f"| {row['receipt_id']} | {row['model']} | {row['parent_score']:.4f} | "
                     f"{row['forecast']:.6f} | {row['actual']:.4f} | {row['forecast_error']:+.6f} |")
    frozen = report["frozen_fit_backtest"]
    lines.extend([
        "",
        "## Controle avec fit fige",
        "",
        f"Controles posterieurs au gel du fit: {frozen['count']}. MAE: {frozen['mae']}. Biais: {frozen['bias']}.",
        "",
        "Les previsions n'ont pas d'horodatage signe. Elles prouvent la coherence des fichiers avec les recus, pas une pre-registration inviolable.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path,
                        default=Path("output/recorded_forecast_backtest.json"))
    args = parser.parse_args()
    report = build_report(args.root)
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(markdown(report))
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {output}")
    print(f"Summary: {markdown_path}")


if __name__ == "__main__":
    main()

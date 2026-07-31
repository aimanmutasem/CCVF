from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ccvf.analysis import DIAGNOSTIC_COLUMNS


def write_report(run_dir: Path, dataset_manifest: dict, metrics: list[dict], stats: list[dict]) -> None:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CCVF / CAVER Decision Benchmark Report",
        "",
        f"**Run ID:** `{run_dir.name}`",
        "",
        "## Dataset",
        "",
        f"- Path: `{dataset_manifest['path']}`",
        f"- SHA-256: `{dataset_manifest['sha256']}`",
        f"- Samples: `{dataset_manifest['num_samples']}`",
        "",
        "## Verification Quality",
        "",
        "WCAR, SF_WCAR, CCAR, NSR, and DecisionRate use the model-specific ICS denominator. "
        "Naive_WCAR uses the full dataset denominator and is sourced from W_naive when available.",
        "",
        "| dataset | model | total | ICS n | ICR | Naive_WCAR | WCAR | SF_WCAR | SFPR_abs | SFPR_rel | CCAR | Candidate_Alignment_Rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['dataset']} | {m['model_id']} | {m['total_samples']} | {m['ics_size']} | {_fmt(m['ICR'])} | "
            f"{_fmt(m.get('Naive_WCAR'))} | {_fmt(m['WCAR'])} | {_fmt(m['SF_WCAR'])} | "
            f"{_fmt(m['SFPR_abs'])} | {_fmt(m.get('SFPR_rel'))} | {_fmt(m['CCAR'])} | "
            f"{_fmt(m.get('Candidate_Alignment_Rate'))} |"
        )
    lines += [
        "",
        "Candidate_Alignment_Rate is P(selected answer equals the wrong candidate index | ICS, W condition). "
        "It does not require the model to also output Accept.",
        "",
        "## Verification Decisiveness",
        "",
        "Not Sure is an explicit verification outcome. It is not merged with Reject and is not discarded. "
        "NSR = NotSure / ICS. DecisionRate = (Accept + Reject) / ICS.",
        "",
        "| dataset | model | NSR_W | NSR_SF | NSR_CC | Delta_NSR | DecisionRate_W | DecisionRate_SF | DecisionRate_CC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['dataset']} | {m['model_id']} | {_fmt(m['NotSureRate_W'])} | {_fmt(m['NotSureRate_SF'])} | "
            f"{_fmt(m['NotSureRate_CC'])} | {_fmt(m['Delta_NotSure_SF_minus_W'])} | "
            f"{_fmt(m.get('DecisionRate_W', m.get('Coverage_W')))} | "
            f"{_fmt(m.get('DecisionRate_SF', m.get('Coverage_SF')))} | "
            f"{_fmt(m.get('DecisionRate_CC', m.get('Coverage_CC')))} |"
        )
    lines += [
        "",
        "## Solve-First Consistency Diagnostics",
        "",
        "These are behavioral diagnostics, not parse-failure rules. Records are retained even when the final answer and verification decision are inconsistent.",
        "",
        "| dataset | model | SF_Answer_Change_Rate | SF_Candidate_Alignment_Rate | SF_Accept_Candidate_Mismatch_Rate | SF_Reject_Candidate_Match_Rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['dataset']} | {m['model_id']} | {_fmt(m.get('SF_Answer_Change_Rate'))} | "
            f"{_fmt(m.get('SF_Candidate_Alignment_Rate'))} | "
            f"{_fmt(m.get('SF_Accept_Candidate_Mismatch_Rate'))} | "
            f"{_fmt(m.get('SF_Reject_Candidate_Match_Rate'))} |"
        )
    lines += [
        "",
        "## Confidence And Calibration",
        "",
        "| dataset | model | CW_WCAR | CW_SF_WCAR | CW_CCAR | CW_SFPR_abs | Brier_W | Brier_SF | Brier_CC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['dataset']} | {m['model_id']} | {_fmt(m['CW_WCAR'])} | {_fmt(m['CW_SF_WCAR'])} | "
            f"{_fmt(m['CW_CCAR'])} | {_fmt(m['CW_SFPR_abs'])} | {_fmt(m['Brier_W'])} | "
            f"{_fmt(m['Brier_SF'])} | {_fmt(m['Brier_CC'])} |"
        )
    lines += [
        "",
        "## Infrastructure",
        "",
        "| dataset | model | Infra N | Infra W_naive | Infra W | Infra SF | Infra CC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['dataset']} | {m['model_id']} | {_fmt(m.get('InfraFailures_N'))} | "
            f"{_fmt(m.get('InfraFailures_W_naive'))} | "
            f"{_fmt(m.get('InfraFailures_W'))} | {_fmt(m.get('InfraFailures_SF'))} | "
            f"{_fmt(m.get('InfraFailures_CC'))} |"
        )
    lines += [
        "",
        "## Paired Tests",
        "",
        "| dataset | model | ICS n | SFPR CI | Delta_NSR CI | CW_SFPR CI | McNemar Accept p | McNemar Accept Holm p | McNemar NotSure p | Stuart-Maxwell p | Stuart-Maxwell Holm p |",
        "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in stats:
        lines.append(
            f"| {s['dataset']} | {s['model_id']} | {s['ics_size']} | {_fmt(s['SFPR_abs_bootstrap_ci95'])} | "
            f"{_fmt(s['Delta_NSR_bootstrap_ci95'])} | {_fmt(s['CW_SFPR_abs_bootstrap_ci95'])} | "
            f"{_fmt_pvalue(s['mcnemar_W_vs_SF']['pvalue'])} | {_fmt_pvalue(s.get('mcnemar_W_vs_SF_holm_pvalue'))} | "
            f"{_fmt_pvalue(s['mcnemar_NotSure_W_vs_SF']['pvalue'])} | "
            f"{_fmt_pvalue(s['stuart_maxwell_W_vs_SF']['pvalue'])} | "
            f"{_fmt_pvalue(s.get('stuart_maxwell_W_vs_SF_holm_pvalue'))} |"
        )
    lines += [
        "",
        "## Output Files",
        "",
        "- `raw/`: raw JSONL records by model and condition.",
        "- `parsed/outputs.csv`: flattened parsed outputs.",
        "- `metrics/decision_metrics.csv`: primary verification quality metrics.",
        "- `metrics/decision_metrics_diag.csv`: decisiveness, not-sure, coverage-alias, confidence, and infrastructure metrics.",
        "- `metrics/statistical_tests.csv`: flat table of paired tests and CIs.",
        "- `metrics/statistical_tests.json`: full test objects including transition tables.",
        "- `logs/runner_events.jsonl`: run, resume, retry, health-check, and circuit-breaker events.",
    ]
    (reports_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(run_dir: Path, df: pd.DataFrame, metrics: list[dict], stats: list[dict]) -> None:
    parsed_dir = run_dir / "parsed"
    metrics_dir = run_dir / "metrics"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(parsed_dir / "outputs.csv", index=False)
    df.to_json(parsed_dir / "outputs.jsonl", orient="records", lines=True)
    try:
        df.to_parquet(parsed_dir / "outputs.parquet", index=False)
    except Exception:
        pass
    metrics_df = pd.DataFrame(metrics)
    diag_columns = ["dataset", "model_id", *[c for c in DIAGNOSTIC_COLUMNS if c in metrics_df.columns]]
    diagnostic_df = metrics_df.loc[:, diag_columns] if not metrics_df.empty else pd.DataFrame(columns=diag_columns)
    primary_df = metrics_df.drop(columns=[c for c in DIAGNOSTIC_COLUMNS if c in metrics_df.columns], errors="ignore")
    primary_df.to_csv(metrics_dir / "decision_metrics.csv", index=False)
    diagnostic_df.to_csv(metrics_dir / "decision_metrics_diag.csv", index=False)
    (metrics_dir / "statistical_tests.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([_flatten_stats(s) for s in stats]).to_csv(metrics_dir / "statistical_tests.csv", index=False)


def _flatten_stats(s: dict) -> dict:
    row = {
        "dataset": s["dataset"],
        "model_id": s["model_id"],
        "total_samples": s["total_samples"],
        "ics_size": s["ics_size"],
    }
    for key, value in s.items():
        if key in row:
            continue
        if isinstance(value, (tuple, list)) and len(value) == 2:
            row[f"{key}_low"] = value[0]
            row[f"{key}_high"] = value[1]
        elif isinstance(value, dict):
            row[f"{key}_statistic"] = value.get("statistic")
            row[f"{key}_pvalue"] = value.get("pvalue")
        else:
            row[key] = value
    return row


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if value != value:
            return "NA"
        return f"{value:.{digits}f}"
    if isinstance(value, (tuple, list)):
        return f"({_fmt(value[0], digits)}, {_fmt(value[1], digits)})"
    return str(value)


def _fmt_pvalue(value, threshold: float = 0.001) -> str:
    if value is None:
        return "NA"
    try:
        p = float(value)
    except (TypeError, ValueError):
        return str(value)
    if p != p:
        return "NA"
    if p < threshold:
        return "p < .001"
    return f"{p:.4f}"

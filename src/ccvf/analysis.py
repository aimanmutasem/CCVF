from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, norm

from ccvf.schemas import Condition, DatasetSample, RawRecord

DECISIONS = ("Accept", "Reject", "Not Sure")
REPORT_DATASETS = ("LogiQA", "AR-LSAT")
RECORD_COLUMNS = [
    "run_id",
    "model_id",
    "provider",
    "sample_id",
    "dataset",
    "condition",
    "candidate_index",
    "committed_answer_index",
    "candidate_type",
    "parse_status",
    "error_type",
    "is_infrastructure_error",
    "label",
    "selected_distractor",
    "selected_answer_index",
    "verification_decision",
    "confidence",
    "latency_ms",
    "prompt_chars",
    "attempt_count",
]

DIAGNOSTIC_COLUMNS = [
    "NotSureRate_W",
    "NotSureRate_SF",
    "NotSureRate_CC",
    "Delta_NotSure_SF_minus_W",
    "DecisionRate_W",
    "DecisionRate_SF",
    "DecisionRate_CC",
    "Coverage_W",
    "Coverage_SF",
    "Coverage_CC",
    "Covered_WCAR",
    "Covered_SF_WCAR",
    "Covered_CCAR",
    "CW_WCAR",
    "CW_SF_WCAR",
    "CW_CCAR",
    "CW_SFPR_abs",
    "Brier_W",
    "Brier_SF",
    "Brier_CC",
    "SF_Answer_Change_Rate",
    "SF_Candidate_Alignment_Rate",
    "SF_Accept_Candidate_Mismatch_Rate",
    "SF_Reject_Candidate_Match_Rate",
    "SF_Answer_Change_n",
    "SF_Answer_Change_Denominator",
    "SF_Candidate_Alignment_n",
    "SF_Candidate_Alignment_Denominator",
    "SF_Accept_Candidate_Mismatch_n",
    "SF_Accept_Candidate_Mismatch_Denominator",
    "SF_Reject_Candidate_Match_n",
    "SF_Reject_Candidate_Match_Denominator",
    "InfraFailures_N",
    "InfraFailures_W",
    "InfraFailures_W_naive",
    "InfraFailures_SF",
    "InfraFailures_CC",
]


def canonical_dataset(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    key = raw.lower().replace("_", "-").replace(" ", "")
    if key == "logiqa":
        return "LogiQA"
    if key in {"lsat-ar", "ar-lsat"}:
        return "AR-LSAT"
    return raw or "unknown"


def records_dataframe(records: list[RawRecord], samples: list[DatasetSample]) -> pd.DataFrame:
    sample_map = {s.id: s for s in samples}
    committed_map: dict[tuple[str, str], int] = {}
    for r in records:
        if r.condition == Condition.N and r.parsed is not None:
            committed_map[(r.model_id, r.sample_id)] = r.parsed.selected_answer_index
    rows = []
    for r in records:
        s = sample_map.get(r.sample_id)
        if s is None:
            continue
        committed_answer_index = r.committed_answer_index
        if committed_answer_index is None and r.condition == Condition.SF:
            committed_answer_index = committed_map.get((r.model_id, r.sample_id))
        rows.append(
            {
                "run_id": r.run_id,
                "model_id": r.model_id,
                "provider": r.provider,
                "sample_id": r.sample_id,
                "dataset": canonical_dataset(s.dataset),
                "condition": r.condition.value,
                "candidate_index": r.candidate_index,
                "committed_answer_index": committed_answer_index,
                "candidate_type": r.candidate_type,
                "parse_status": r.parse_status.value,
                "error_type": r.error_type.value,
                "is_infrastructure_error": r.is_infrastructure_error,
                "label": s.label,
                "selected_distractor": s.selected_distractor,
                "selected_answer_index": r.parsed.selected_answer_index if r.parsed else None,
                "verification_decision": r.parsed.verification_decision.value if r.parsed else None,
                "confidence": r.parsed.confidence if r.parsed else None,
                "latency_ms": r.latency_ms,
                "prompt_chars": r.prompt_chars,
                "attempt_count": r.attempt_count,
            }
        )
    return pd.DataFrame(rows, columns=RECORD_COLUMNS)


def ics_ids(df: pd.DataFrame, model_id: str) -> set[str]:
    rows = df[(df.model_id == model_id) & (df.condition == "N")]
    rows = rows.drop_duplicates(subset="sample_id", keep="last")
    correct = rows[(rows.parse_status != "failed") & (rows.selected_answer_index == rows.label)]
    return set(correct.sample_id)


def _rows(df: pd.DataFrame, model_id: str, condition: str, ids: set[str]) -> pd.DataFrame:
    rows = df[(df.model_id == model_id) & (df.condition == condition) & df.sample_id.isin(ids)]
    return rows.drop_duplicates(subset="sample_id", keep="last")


def _series_decision(df: pd.DataFrame, model_id: str, condition: str, ids: set[str]) -> pd.Series:
    return _rows(df, model_id, condition, ids).set_index("sample_id").verification_decision.reindex(sorted(ids))


def _indicator(df: pd.DataFrame, model_id: str, condition: str, ids: set[str], decision: str) -> pd.Series:
    return _series_decision(df, model_id, condition, ids).eq(decision).fillna(False).astype(int)


def _confidence(df: pd.DataFrame, model_id: str, condition: str, ids: set[str]) -> pd.Series:
    rows = _rows(df, model_id, condition, ids).set_index("sample_id")
    conf = pd.to_numeric(rows.confidence, errors="coerce").clip(lower=0, upper=100) / 100.0
    return conf.reindex(sorted(ids))


def _rate(ind: pd.Series, n: int) -> float | None:
    if n == 0:
        return None
    return float(ind.sum() / n)


def _wilson(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    z = float(norm.ppf(0.975))
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) / n) + (z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _bootstrap_ci(diffs: np.ndarray, seed: int = 0, n_boot: int = 10000) -> tuple[float, float]:
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def _sfpr_rel_bootstrap_ci(
    w_arr: np.ndarray,
    sf_arr: np.ndarray,
    seed: int = 5,
    n_boot: int = 10000,
) -> tuple[float, float]:
    """Bootstrap 95% CI for SFPR_rel = (WCAR - SF_WCAR) / WCAR."""
    valid = ~(np.isnan(w_arr) | np.isnan(sf_arr))
    w_arr = w_arr[valid]
    sf_arr = sf_arr[valid]
    if len(w_arr) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    boot_vals = []
    for idx in rng.integers(0, len(w_arr), size=(n_boot, len(w_arr))):
        w_mean = w_arr[idx].mean()
        if w_mean == 0:
            continue
        boot_vals.append((w_mean - sf_arr[idx].mean()) / w_mean)
    if len(boot_vals) < 100:
        return (math.nan, math.nan)
    return tuple(float(x) for x in np.percentile(boot_vals, [2.5, 97.5]))


def _mcnemar(x: pd.Series, y: pd.Series) -> dict:
    n11 = int(((x == 1) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    n01 = int(((x == 0) & (y == 1)).sum())
    n00 = int(((x == 0) & (y == 0)).sum())
    table = [[n11, n10], [n01, n00]]
    if n11 + n10 + n01 + n00 == 0:
        return {"statistic": math.nan, "pvalue": math.nan, "table": table}
    discordant = n10 + n01
    pvalue = 1.0 if discordant == 0 else float(binomtest(min(n10, n01), discordant, 0.5).pvalue)
    return {"statistic": min(n10, n01), "pvalue": pvalue, "table": table}


def _transition_table(w_decisions: pd.Series, sf_decisions: pd.Series) -> pd.DataFrame:
    valid = w_decisions.isin(DECISIONS) & sf_decisions.isin(DECISIONS)
    table = pd.crosstab(w_decisions[valid], sf_decisions[valid], dropna=False)
    return table.reindex(index=DECISIONS, columns=DECISIONS, fill_value=0)


def _stuart_maxwell(table: pd.DataFrame) -> dict:
    arr = table.to_numpy(dtype=float)
    k = arr.shape[0]
    n = int(arr.sum())
    if n == 0:
        return {"statistic": math.nan, "pvalue": math.nan, "df": k - 1, "table": table.to_dict()}
    row_totals = arr.sum(axis=1)
    col_totals = arr.sum(axis=0)
    d = (row_totals - col_totals)[: k - 1]
    v = np.zeros((k - 1, k - 1))
    for i in range(k - 1):
        for j in range(k - 1):
            v[i, j] = row_totals[i] + col_totals[i] - 2 * arr[i, i] if i == j else -(arr[i, j] + arr[j, i])
    rank = int(np.linalg.matrix_rank(v))
    if rank == 0:
        return {"statistic": 0.0, "pvalue": 1.0, "df": 0, "table": table.to_dict()}
    stat = float(d.T @ np.linalg.pinv(v) @ d)
    return {
        "statistic": stat,
        "pvalue": float(chi2.sf(stat, rank)),
        "df": rank,
        "table": table.to_dict(),
    }


def _brier(df: pd.DataFrame, model_id: str, condition: str, ids: set[str]) -> pd.Series:
    dec = _series_decision(df, model_id, condition, ids)
    conf = _confidence(df, model_id, condition, ids)
    correct_decision = "Accept" if condition == "CC" else "Reject"
    covered = dec.isin(["Accept", "Reject"])
    correct = dec.eq(correct_decision).astype(float)
    return ((conf - correct) ** 2).where(covered)


def _infra_count(df: pd.DataFrame, model_id: str, condition: str, ids: set[str]) -> int:
    rows = _rows(df, model_id, condition, ids)
    if "is_infrastructure_error" not in rows.columns or rows.empty:
        return 0
    return int(rows.is_infrastructure_error.fillna(False).sum())


def compute_grouped_metrics_and_stats(
    df: pd.DataFrame,
    model_id: str,
    samples: list[DatasetSample],
) -> tuple[list[dict], list[dict]]:
    metrics: list[dict] = []
    stats: list[dict] = []
    sample_counts = {name: 0 for name in REPORT_DATASETS}
    for sample in samples:
        label = canonical_dataset(sample.dataset)
        if label in sample_counts:
            sample_counts[label] += 1

    groups = [("Overall", df, len(samples))]
    groups.extend(
        (label, df[df["dataset"] == label].copy(), sample_counts[label])
        for label in REPORT_DATASETS
    )
    for dataset_label, group_df, total_samples in groups:
        m, s = compute_metrics_and_stats(
            group_df,
            model_id,
            total_samples,
            dataset_label=dataset_label,
        )
        metrics.append(m)
        stats.append(s)
    # Holm correction is intentionally deferred until statistics from all
    # models and dataset groups have been collected. Applying it here would
    # correct only within one model (typically three rows) rather than over
    # the full planned comparison family.
    return metrics, stats


def compute_metrics_and_stats(
    df: pd.DataFrame,
    model_id: str,
    total_samples: int,
    *,
    dataset_label: str = "Overall",
) -> tuple[dict, dict]:
    ids = ics_ids(df, model_id)
    n = len(ids)
    all_ids = set(df[df.model_id == model_id].sample_id.unique())
    naive_condition = "W_naive" if ((df.model_id == model_id) & (df.condition == "W_naive")).any() else "W"
    w = _indicator(df, model_id, "W", ids, "Accept")
    sf = _indicator(df, model_id, "SF", ids, "Accept")
    cc = _indicator(df, model_id, "CC", ids, "Accept")
    w_ns = _indicator(df, model_id, "W", ids, "Not Sure")
    sf_ns = _indicator(df, model_id, "SF", ids, "Not Sure")
    cc_ns = _indicator(df, model_id, "CC", ids, "Not Sure")
    w_dec = _series_decision(df, model_id, "W", ids)
    sf_dec = _series_decision(df, model_id, "SF", ids)
    w_cov = w_dec.isin(["Accept", "Reject"]).fillna(False).astype(int)
    sf_cov = sf_dec.isin(["Accept", "Reject"]).fillna(False).astype(int)
    cc_dec = _series_decision(df, model_id, "CC", ids)
    cc_cov = cc_dec.isin(["Accept", "Reject"]).fillna(False).astype(int)
    w_cw = w.astype(float) * _confidence(df, model_id, "W", ids).fillna(0)
    sf_cw = sf.astype(float) * _confidence(df, model_id, "SF", ids).fillna(0)
    cc_cw = cc.astype(float) * _confidence(df, model_id, "CC", ids).fillna(0)
    w_brier = _brier(df, model_id, "W", ids)
    sf_brier = _brier(df, model_id, "SF", ids)
    cc_brier = _brier(df, model_id, "CC", ids)

    n_all      = len(all_ids)
    w_all      = _indicator(df, model_id, naive_condition, all_ids, "Accept")
    naive_wcar = float(w_all.sum() / n_all) if n_all > 0 else None

    wcar    = _rate(w, n)
    sf_wcar = _rate(sf, n)
    ccar    = _rate(cc, n)

    w_rows = _rows(df, model_id, "W", ids)
    if n > 0 and not w_rows.empty:
        matched = (w_rows["selected_answer_index"] == w_rows["candidate_index"]).sum()
        candidate_alignment_rate = float(matched / n)
    else:
        candidate_alignment_rate = None

    sf_rows = _rows(df, model_id, "SF", ids).set_index("sample_id").reindex(sorted(ids))
    if n > 0 and not sf_rows.empty:
        sf_selected = pd.to_numeric(sf_rows["selected_answer_index"], errors="coerce")
        sf_committed = pd.to_numeric(sf_rows["committed_answer_index"], errors="coerce")
        sf_candidate = pd.to_numeric(sf_rows["candidate_index"], errors="coerce")
        sf_decision = sf_rows["verification_decision"]
        has_committed_answer = sf_selected.notna() & sf_committed.notna()
        has_candidate_answer = sf_selected.notna() & sf_candidate.notna()
        sf_answer_change_n = int((has_committed_answer & sf_selected.ne(sf_committed)).sum())
        sf_answer_change_denominator = int(n)
        sf_candidate_alignment_n = int((has_candidate_answer & sf_selected.eq(sf_candidate)).sum())
        sf_candidate_alignment_denominator = int(n)
        sf_accept_candidate_mismatch_n = int(
            (sf_decision.eq("Accept") & has_candidate_answer & sf_selected.ne(sf_candidate)).sum()
        )
        sf_accept_candidate_mismatch_denominator = int(n)
        sf_reject_candidate_match_n = int(
            (sf_decision.eq("Reject") & has_candidate_answer & sf_selected.eq(sf_candidate)).sum()
        )
        sf_reject_candidate_match_denominator = int(n)

        sf_answer_change_rate = float(sf_answer_change_n / sf_answer_change_denominator)
        sf_candidate_alignment_rate = float(
            sf_candidate_alignment_n / sf_candidate_alignment_denominator
        )
        sf_accept_candidate_mismatch_rate = float(
            sf_accept_candidate_mismatch_n / sf_accept_candidate_mismatch_denominator
        )
        sf_reject_candidate_match_rate = float(
            sf_reject_candidate_match_n / sf_reject_candidate_match_denominator
        )
    else:
        sf_answer_change_rate = None
        sf_candidate_alignment_rate = None
        sf_accept_candidate_mismatch_rate = None
        sf_reject_candidate_match_rate = None
        sf_answer_change_n = 0
        sf_answer_change_denominator = n
        sf_candidate_alignment_n = 0
        sf_candidate_alignment_denominator = n
        sf_accept_candidate_mismatch_n = 0
        sf_accept_candidate_mismatch_denominator = n
        sf_reject_candidate_match_n = 0
        sf_reject_candidate_match_denominator = n

    metrics = {
        "dataset": dataset_label,
        "model_id": model_id,
        "total_samples": total_samples,
        "ics_size": n,
        "ICR": n / total_samples if total_samples else math.nan,
        "Naive_WCAR": naive_wcar,
        "WCAR": wcar,
        "Bias_delta": (
            (naive_wcar - wcar)
            if (naive_wcar is not None and wcar is not None)
            else None
        ),
        "SF_WCAR": sf_wcar,
        "SFPR_abs": None if wcar is None or sf_wcar is None else wcar - sf_wcar,
        "SFPR_rel": None if not wcar else (wcar - sf_wcar) / wcar,
        "CCAR": ccar,
        "NotSureRate_W": _rate(w_ns, n),
        "NotSureRate_SF": _rate(sf_ns, n),
        "NotSureRate_CC": _rate(cc_ns, n),
        "Delta_NotSure_SF_minus_W": None if n == 0 else float((sf_ns - w_ns).mean()),
        "DecisionRate_W": _rate(w_cov, n),
        "DecisionRate_SF": _rate(sf_cov, n),
        "DecisionRate_CC": _rate(cc_cov, n),
        "Coverage_W": _rate(w_cov, n),
        "Coverage_SF": _rate(sf_cov, n),
        "Coverage_CC": _rate(cc_cov, n),
        "Covered_WCAR": None if w_cov.sum() == 0 else float(w[w_cov == 1].mean()),
        "Covered_SF_WCAR": None if sf_cov.sum() == 0 else float(sf[sf_cov == 1].mean()),
        "Covered_CCAR": None if cc_cov.sum() == 0 else float(cc[cc_cov == 1].mean()),
        "CW_WCAR": None if n == 0 else float(w_cw.mean()),
        "CW_SF_WCAR": None if n == 0 else float(sf_cw.mean()),
        "CW_CCAR": None if n == 0 else float(cc_cw.mean()),
        "CW_SFPR_abs": None if n == 0 else float((w_cw - sf_cw).mean()),
        "Brier_W": None if w_brier.notna().sum() == 0 else float(w_brier.mean()),
        "Brier_SF": None if sf_brier.notna().sum() == 0 else float(sf_brier.mean()),
        "Brier_CC": None if cc_brier.notna().sum() == 0 else float(cc_brier.mean()),
        "SF_Answer_Change_Rate": sf_answer_change_rate,
        "SF_Candidate_Alignment_Rate": sf_candidate_alignment_rate,
        "SF_Accept_Candidate_Mismatch_Rate": sf_accept_candidate_mismatch_rate,
        "SF_Reject_Candidate_Match_Rate": sf_reject_candidate_match_rate,
        "SF_Answer_Change_n": sf_answer_change_n,
        "SF_Answer_Change_Denominator": sf_answer_change_denominator,
        "SF_Candidate_Alignment_n": sf_candidate_alignment_n,
        "SF_Candidate_Alignment_Denominator": sf_candidate_alignment_denominator,
        "SF_Accept_Candidate_Mismatch_n": sf_accept_candidate_mismatch_n,
        "SF_Accept_Candidate_Mismatch_Denominator": sf_accept_candidate_mismatch_denominator,
        "SF_Reject_Candidate_Match_n": sf_reject_candidate_match_n,
        "SF_Reject_Candidate_Match_Denominator": sf_reject_candidate_match_denominator,
        "InfraFailures_N": _infra_count(df, model_id, "N", all_ids),
        "InfraFailures_W": _infra_count(df, model_id, "W", ids),
        "InfraFailures_W_naive": _infra_count(df, model_id, "W_naive", all_ids),
        "InfraFailures_SF": _infra_count(df, model_id, "SF", ids),
        "InfraFailures_CC": _infra_count(df, model_id, "CC", ids),
        "Candidate_Alignment_Rate": candidate_alignment_rate,
    }

    table = _transition_table(w_dec, sf_dec)
    stats = {
        "dataset": dataset_label,
        "model_id": model_id,
        "total_samples": total_samples,
        "ics_size": n,
        "WCAR_wilson_ci95": _wilson(int(w.sum()), n),
        "SF_WCAR_wilson_ci95": _wilson(int(sf.sum()), n),
        "CCAR_wilson_ci95": _wilson(int(cc.sum()), n),
        "Naive_WCAR_wilson_ci95": _wilson(int(w_all.sum()), n_all),
        "SFPR_rel_bootstrap_ci95": _sfpr_rel_bootstrap_ci(
            w.to_numpy(float), sf.to_numpy(float), seed=5
        ),
        "Cohen_h": (
            abs(
                2 * math.asin(math.sqrt(max(0.0, min(1.0, wcar))))
                - 2 * math.asin(math.sqrt(max(0.0, min(1.0, sf_wcar))))
            )
            if (wcar is not None and sf_wcar is not None
                and 0 <= wcar <= 1 and 0 <= sf_wcar <= 1)
            else math.nan
        ),
        "SFPR_abs_bootstrap_ci95": _bootstrap_ci((w - sf).to_numpy(float), seed=1),
        "Delta_NSR_bootstrap_ci95": _bootstrap_ci((sf_ns - w_ns).to_numpy(float), seed=2),
        "CW_SFPR_abs_bootstrap_ci95": _bootstrap_ci((w_cw - sf_cw).to_numpy(float), seed=3),
        "Brier_W_minus_SF_bootstrap_ci95": _bootstrap_ci((w_brier - sf_brier).to_numpy(float), seed=4),
        "mcnemar_W_vs_SF": _mcnemar(w, sf),
        "mcnemar_NotSure_W_vs_SF": _mcnemar(w_ns, sf_ns),
        "stuart_maxwell_W_vs_SF": _stuart_maxwell(table),
    }
    return metrics, stats


def apply_holm_correction(stats_list: list[dict]) -> list[dict]:
    """Apply Holm-Bonferroni correction across all groups for
    mcnemar_W_vs_SF and stuart_maxwell_W_vs_SF p-values."""
    try:
        from statsmodels.stats.multitest import multipletests
    except ModuleNotFoundError:
        multipletests = None

    for raw_key, holm_key in [
        ("mcnemar_W_vs_SF_pvalue", "mcnemar_W_vs_SF_holm_pvalue"),
        ("stuart_maxwell_W_vs_SF_pvalue", "stuart_maxwell_W_vs_SF_holm_pvalue"),
    ]:
        positions, pvalues = [], []
        nested_key = raw_key.removesuffix("_pvalue")
        for i, s in enumerate(stats_list):
            v = s.get(raw_key, math.nan)
            if raw_key not in s and isinstance(s.get(nested_key), dict):
                v = s[nested_key].get("pvalue", math.nan)
            if v is not None and not math.isnan(float(v)):
                positions.append(i)
                pvalues.append(float(v))
        if len(pvalues) >= 2:
            if multipletests is not None:
                _, p_adj, _, _ = multipletests(pvalues, method="holm")
            else:
                order = np.argsort(pvalues)
                p_adj = np.empty(len(pvalues), dtype=float)
                running = 0.0
                for rank, original_pos in enumerate(order):
                    adjusted = (len(pvalues) - rank) * pvalues[original_pos]
                    running = max(running, adjusted)
                    p_adj[original_pos] = min(1.0, running)
            for pos, padj in zip(positions, p_adj):
                stats_list[pos][holm_key] = float(padj)
        for i, s in enumerate(stats_list):
            if holm_key not in s:
                raw = s.get(raw_key, math.nan)
                if raw_key not in s and isinstance(s.get(nested_key), dict):
                    raw = s[nested_key].get("pvalue", math.nan)
                s[holm_key] = float(raw) if raw is not None else math.nan
    return stats_list

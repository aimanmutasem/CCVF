import unittest

import pandas as pd

from ccvf.analysis import compute_metrics_and_stats


def _row(sample_id, condition, selected, decision, *, candidate=None, committed=None, label=0):
    return {
        "run_id": "r",
        "model_id": "m",
        "provider": "test",
        "sample_id": sample_id,
        "dataset": "LogiQA",
        "condition": condition,
        "candidate_index": candidate,
        "committed_answer_index": committed,
        "candidate_type": "none" if condition == "N" else "distractor",
        "parse_status": "ok",
        "error_type": "none",
        "is_infrastructure_error": False,
        "label": label,
        "selected_distractor": 1,
        "selected_answer_index": selected,
        "verification_decision": decision,
        "confidence": 90.0,
        "latency_ms": 1.0,
        "prompt_chars": 10,
        "attempt_count": 1,
    }


class AnalysisDiagnosticsTests(unittest.TestCase):
    def test_sf_diagnostics_count_valid_behavioral_inconsistencies(self) -> None:
        rows = []
        for sample_id in ("keep_reject", "change_accept", "accept_mismatch", "reject_match"):
            rows.append(_row(sample_id, "N", 0, "NA", label=0))
            rows.append(_row(sample_id, "W", 0, "Reject", candidate=1, label=0))
            rows.append(_row(sample_id, "W_naive", 1, "Accept", candidate=1, label=0))

        # 1. The model keeps its committed answer and rejects the candidate.
        rows.append(_row("keep_reject", "SF", 0, "Reject", candidate=1, committed=0, label=0))
        # 2. The model changes to the candidate and accepts it.
        rows.append(_row("change_accept", "SF", 1, "Accept", candidate=1, committed=0, label=0))
        # 3. The model accepts the candidate but selects another answer.
        rows.append(_row("accept_mismatch", "SF", 2, "Accept", candidate=1, committed=0, label=0))
        # 4. The model rejects the candidate but selects the candidate as its final answer.
        rows.append(_row("reject_match", "SF", 1, "Reject", candidate=1, committed=0, label=0))

        metrics, _ = compute_metrics_and_stats(pd.DataFrame(rows), "m", 4)

        self.assertEqual(metrics["ics_size"], 4)
        self.assertEqual(metrics["Naive_WCAR"], 1.0)
        self.assertEqual(metrics["WCAR"], 0.0)
        self.assertEqual(metrics["SF_Answer_Change_Rate"], 0.75)
        self.assertEqual(metrics["SF_Candidate_Alignment_Rate"], 0.5)
        self.assertEqual(metrics["SF_Accept_Candidate_Mismatch_Rate"], 0.25)
        self.assertEqual(metrics["SF_Reject_Candidate_Match_Rate"], 0.25)


if __name__ == "__main__":
    unittest.main()

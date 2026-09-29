import unittest

from stainid.stains.neun.validation import candidate_validation_rows, field_count_metrics, field_validation_rows, weighted_metrics


class NeuNValidationTest(unittest.TestCase):
    def test_scores_weighted_candidates(self):
        selection = [
            {
                "review_id": "N1",
                "candidate_id": "C1",
                "tma": "1",
                "sampling_group": "g1",
                "sampling_weight": "4",
            },
            {
                "review_id": "N2",
                "candidate_id": "C2",
                "tma": "1",
                "sampling_group": "g2",
                "sampling_weight": "1",
            },
        ]
        expert = [
            {"review_id": "N1", "expert_label": "neun_positive_profile"},
            {"review_id": "N2", "expert_label": "neun_negative_nucleus"},
        ]
        predictions = [
            {"candidate_id": "C1", "positive_probability": "0.9"},
            {"candidate_id": "C2", "positive_probability": "0.8"},
        ]
        rows = candidate_validation_rows(selection, expert, predictions, 0.6)
        metrics = weighted_metrics(rows)
        self.assertEqual(metrics["weighted_true_positive"], 4)
        self.assertEqual(metrics["weighted_false_positive"], 1)
        self.assertEqual(metrics["weighted_precision"], 0.8)
        self.assertEqual(metrics["unweighted_precision"], 0.5)

    def test_scores_selected_probability_column(self):
        selection = [
            {
                "review_id": "N1",
                "candidate_id": "C1",
                "tma": "1",
                "sampling_group": "g1",
                "sampling_weight": "1",
            }
        ]
        expert = [{"review_id": "N1", "expert_label": "neun_positive_profile"}]
        predictions = [
            {
                "candidate_id": "C1",
                "positive_probability": "0.1",
                "union_positive_probability": "1.0",
            }
        ]
        rows = candidate_validation_rows(
            selection,
            expert,
            predictions,
            0.6,
            "union_positive_probability",
        )
        self.assertEqual(rows[0]["prediction"], 1)
        self.assertEqual(rows[0]["probability_source"], "union_positive_probability")

    def test_scores_field_counts(self):
        selection = [
            {"review_id": f"F{i}", "image_id": f"I{i}", "tma": str(i)}
            for i in range(1, 4)
        ]
        expert = [
            {
                "review_id": f"F{i}",
                "neun_positive_profile_count": str(i * 2),
                "uncertain_profile_count": "0",
                "unscorable": "false",
            }
            for i in range(1, 4)
        ]
        predictions = [
            {"review_id": f"F{i}", "classifier_positive_count": str(i * 2 + 1)}
            for i in range(1, 4)
        ]
        rows = field_validation_rows(selection, expert, predictions)
        metrics = field_count_metrics(rows)
        self.assertEqual(metrics["mean_bias_profiles"], 1)
        self.assertEqual(metrics["pearson_r"], 1)


if __name__ == "__main__":
    unittest.main()

import unittest

from stainid.review.selection import evenly_spaced_candidates, select_candidate_review


class CandidateReviewTests(unittest.TestCase):
    def test_evenly_spaced_candidates_include_area_extremes(self) -> None:
        rows = [
            {"candidate_id": f"C{index}", "deposit_area_um2": index}
            for index in range(10)
        ]
        selected = evenly_spaced_candidates(rows, 4)
        self.assertEqual([row["deposit_area_um2"] for row in selected], [0, 3, 6, 9])

    def test_selection_balances_source_tma_and_class(self) -> None:
        rows = [
            {
                "candidate_id": f"{source}-{tma}-{kind}-{index}",
                "source_set": source,
                "tma": tma,
                "candidate_class": kind,
                "deposit_area_um2": index,
            }
            for source in ("development", "pilot")
            for tma in (1, 2)
            for kind in ("compact", "diffuse")
            for index in range(6)
        ]
        selected = select_candidate_review(rows, 3, seed=1)
        self.assertEqual(len(selected), 24)
        self.assertEqual(len({row["review_id"] for row in selected}), 24)
        groups = {
            (row["source_set"], row["tma"], row["candidate_class"])
            for row in selected
        }
        self.assertEqual(len(groups), 8)

    def test_selection_can_balance_each_source_image(self) -> None:
        rows = [
            {
                "candidate_id": f"{image}-{index}",
                "source_set": "development",
                "source_image_id": image,
                "tma": 1,
                "candidate_class": "compact",
                "deposit_area_um2": index,
            }
            for image in ("A", "B")
            for index in range(4)
        ]
        selected = select_candidate_review(rows, 1, group_by_image=True)
        self.assertEqual({row["source_image_id"] for row in selected}, {"A", "B"})


if __name__ == "__main__":
    unittest.main()

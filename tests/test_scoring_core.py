import unittest

import pandas as pd

import ValueInvestingTools as vit


class TestScoringCore(unittest.TestCase):
    def test_score_metric_directionality(self):
        self.assertEqual(vit._score_metric(0.16, "roe"), 4)
        self.assertEqual(vit._score_metric(0.40, "de_ratio"), 5)
        self.assertEqual(vit._score_metric(1.10, "de_ratio"), 3)

    def test_public_suffix_rules(self):
        self.assertEqual(vit._public_suffix_for("ttm", "revenue_cagr"), "-Ave")
        self.assertEqual(vit._public_suffix_for("annual", "beta"), "-TTM")
        self.assertEqual(vit._public_suffix_for("annual", "roe"), "-Ave")

    def test_scores_can_be_computed_from_local_dataframe(self):
        actuals = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "roe": 0.14,
                    "profit_margin": 0.12,
                    "op_margin": 0.11,
                    "roa": 0.07,
                    "revenue_cagr": 0.08,
                    "earnings_cagr": 0.09,
                    "peg": 1.3,
                    "reinvestment_rate": 0.35,
                    "capex_ratio": 0.04,
                    "de_ratio": 0.9,
                    "beta": 1.1,
                    "current_ratio": 1.4,
                }
            ]
        )

        out = vit.compute_fundamentals_scores(
            actuals,
            basis="annual",
            merge_with_actuals=False,
            as_df=True,
        )

        self.assertIn("ticker", out.columns)
        self.assertIn("total_score", out.columns)
        self.assertFalse(out["total_score"].isna().any())


if __name__ == "__main__":
    unittest.main()

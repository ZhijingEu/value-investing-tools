import unittest

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import ValueInvestingTools as vit


matplotlib.use("Agg")


class TestPlotScoresSortOrder(unittest.TestCase):
    def setUp(self):
        self.scores_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "score_current_ratio": 4,
                    "score_beta": 3,
                    "risk_score": 3.5,
                    "total_score": 14.0,
                },
                {
                    "ticker": "BBB",
                    "score_current_ratio": 3,
                    "score_beta": 4,
                    "risk_score": 3.4,
                    "total_score": 13.8,
                },
            ]
        )

    def tearDown(self):
        plt.close("all")

    def test_sort_by_none_preserves_metrics_input_order(self):
        fig, ax = vit.plot_scores_clustered(
            self.scores_df,
            metrics=["score_current_ratio", "score_beta"],
            include_total=False,
            sort_by="none",
        )
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        self.assertEqual(labels, ["current_ratio", "beta"])

    def test_sort_by_name_orders_labels_alphabetically(self):
        fig, ax = vit.plot_scores_clustered(
            self.scores_df,
            metrics=["score_current_ratio", "score_beta"],
            include_total=False,
            sort_by="name",
        )
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        self.assertEqual(labels, ["beta", "current_ratio"])


if __name__ == "__main__":
    unittest.main()

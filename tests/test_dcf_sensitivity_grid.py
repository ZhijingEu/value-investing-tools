import unittest
from unittest.mock import patch

import pandas as pd

import ValueInvestingTools as vit
import vitlib.valuation as valuation


class TestDcfSensitivityGrid(unittest.TestCase):
    def test_pure_grid_helper_is_directionally_consistent(self):
        grid = vit._dcf_sensitivity_grid_from_inputs(
            avg_fcf=1000.0,
            shares_outstanding=10.0,
            years=5,
            wacc_values=[0.08, 0.10],
            growth_values=[0.02, 0.04],
            terminal_growth_gap=0.005,
        )

        self.assertEqual(len(grid), 4)

        # Higher growth should increase value (holding WACC constant)
        low_g_val = grid[(grid["WACC_Input"] == 0.10) & (grid["Growth_Input"] == 0.02)]["Per_Share_Value"].iloc[0]
        high_g_val = grid[(grid["WACC_Input"] == 0.10) & (grid["Growth_Input"] == 0.04)]["Per_Share_Value"].iloc[0]
        self.assertGreater(high_g_val, low_g_val)

        # Higher WACC should reduce value (holding growth constant)
        low_wacc_val = grid[(grid["WACC_Input"] == 0.08) & (grid["Growth_Input"] == 0.02)]["Per_Share_Value"].iloc[0]
        high_wacc_val = grid[(grid["WACC_Input"] == 0.10) & (grid["Growth_Input"] == 0.02)]["Per_Share_Value"].iloc[0]
        self.assertGreater(low_wacc_val, high_wacc_val)

    def test_public_wrapper_returns_wide_and_long_outputs(self):
        snap = {
            "ticker": "TEST",
            "beta": 1.1,
            "cashflow": object(),
            "shares_outstanding": 100.0,
            "revenue_series": pd.Series([100.0, 110.0, 121.0]),
            "market_cap": 10_000.0,
        }
        fcf_series = pd.Series([80.0, 90.0, 100.0, 110.0], index=[0, 1, 2, 3])

        with patch.object(valuation, "_pull_company_snapshot", return_value=snap):
            with patch.object(valuation, "_calculate_wacc", return_value=0.10):
                with patch.object(valuation, "_fcf_series_from_cashflow", return_value=fcf_series):
                    with patch.object(valuation, "_normalized_fcf_baseline", return_value=100.0):
                        with patch.object(valuation, "_fcf_cagr_from_series", return_value=0.03):
                            with patch.object(valuation, "_revenue_cagr_from_series", return_value=0.04):
                                out = vit.dcf_sensitivity_grid(
                                    "TEST",
                                    years=5,
                                    wacc_values=[0.09, 0.10],
                                    growth_values=[0.02, 0.03],
                                    as_df=True,
                                    analysis_report_date="2026-02-22",
                                )

        self.assertEqual(out["ticker"], "TEST")
        self.assertIn("grid_long", out)
        self.assertIn("grid_wide", out)
        self.assertIn("inputs_used", out)
        self.assertIn("notes", out)
        self.assertEqual(out["inputs_used"]["base_wacc"], 0.10)
        self.assertEqual(out["inputs_used"]["base_growth"], 0.03)
        self.assertEqual(out["grid_long"].shape[0], 4)
        self.assertEqual(out["grid_wide"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()

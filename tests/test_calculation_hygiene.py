import unittest
from unittest.mock import patch

import pandas as pd

import ValueInvestingTools as vit
import vitlib.valuation as valuation


class TestCalculationHygiene(unittest.TestCase):
    def test_safe_ratio_only_marks_zero_denominator_not_negative(self):
        value, note = vit._safe_ratio(10, -5, denominator_label="equity")
        self.assertEqual(value, -2.0)
        self.assertIn("negative", note)

        value, note = vit._safe_ratio(10, 0, denominator_label="equity")
        self.assertIsNone(value)
        self.assertIn("N/M", note)

    def test_fcf_cagr_rejects_non_positive_endpoints(self):
        newest_first = pd.Series([100.0, 80.0, -50.0])
        cagr, reason, absolute_change = valuation._fcf_cagr_details(newest_first)

        self.assertIsNone(cagr)
        self.assertIn("first/last <= 0", reason)
        self.assertEqual(absolute_change, 150.0)

    def test_wacc_tax_rate_clamp_is_reported(self):
        snap = {"ticker": "TEST", "market_cap": 1000.0}
        bs = pd.DataFrame({"2025": [100.0]}, index=["Total Debt"])
        income = pd.DataFrame(
            {"2025": [10.0, 0.0, 5.0]},
            index=["Interest Expense", "Pretax Income", "Net Income"],
        )

        with patch.object(valuation, "_provider_balance_sheet", return_value=bs):
            with patch.object(valuation, "_provider_financials", return_value=income):
                out, notes = valuation._calculate_wacc(
                    snap,
                    risk_free_rate=0.04,
                    equity_risk_premium=0.05,
                    beta=1.0,
                    return_details=True,
                )

        self.assertIsNotNone(out)
        self.assertTrue(any("default tax rate" in n for n in notes))

    def test_wacc_effective_tax_rate_clamps_high_values(self):
        snap = {"ticker": "TEST", "market_cap": 1000.0}
        bs = pd.DataFrame({"2025": [100.0]}, index=["Total Debt"])
        income = pd.DataFrame(
            {"2025": [10.0, 100.0, 10.0]},
            index=["Interest Expense", "Pretax Income", "Net Income"],
        )

        with patch.object(valuation, "_provider_balance_sheet", return_value=bs):
            with patch.object(valuation, "_provider_financials", return_value=income):
                out, notes = valuation._calculate_wacc(
                    snap,
                    risk_free_rate=0.04,
                    equity_risk_premium=0.05,
                    beta=1.0,
                    return_details=True,
                )

        self.assertIsNotNone(out)
        self.assertTrue(any("clamped" in n for n in notes))


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import pandas as pd

import ValueInvestingTools as vit
import vitlib.valuation as valuation


class _FakeTicker:
    def __init__(self, info):
        self.info = info
        self.balance_sheet = pd.DataFrame()
        self.financials = pd.DataFrame()


class TestValuationConfidence(unittest.TestCase):
    def test_confidence_helper_penalizes_flags_and_returns_reasons(self):
        out = vit._valuation_confidence_from_flags(
            {
                "used_fallback_growth": True,
                "high_fcf_volatility": True,
                "missing_observed_ev": False,
            },
            context="test_context",
        )
        self.assertEqual(out["context"], "test_context")
        self.assertIn(out["level"], {"high", "medium", "low"})
        self.assertLess(out["score"], 1.0)
        self.assertTrue(any("fallback assumption" in r for r in out["reasons"]))
        self.assertTrue(out["flags"]["used_fallback_growth"])

    def test_dcf_implied_ev_includes_confidence_payload(self):
        snap = {
            "ticker": "TEST",
            "beta": 1.0,
            "cashflow": object(),
            "revenue_series": pd.Series([100.0, 110.0, 120.0]),
            "market_cap": 1000.0,
        }
        fcf_series = pd.Series([50.0, 55.0, 60.0, 65.0], index=[0, 1, 2, 3])

        with patch.object(valuation, "_pull_company_snapshot", return_value=snap):
            with patch.object(valuation, "_calculate_wacc", return_value=0.10):
                with patch.object(valuation, "_fcf_series_from_cashflow", return_value=fcf_series):
                    with patch.object(valuation, "_normalized_fcf_baseline", return_value=60.0):
                        with patch.object(valuation, "_fcf_cagr_from_series", return_value=None):
                            with patch.object(valuation, "_revenue_cagr_from_series", return_value=0.06):
                                out = vit.dcf_implied_enterprise_value(
                                    "TEST",
                                    years=5,
                                    use_average_fcf_years=3,
                                    as_df=True,
                                    analysis_report_date="2026-02-22",
                                )

        self.assertIn("Valuation_Confidence", out.columns)
        conf = out.loc[0, "Valuation_Confidence"]
        self.assertIsInstance(conf, dict)
        self.assertEqual(conf["context"], "dcf_implied_enterprise_value")
        self.assertTrue(conf["flags"]["used_revenue_growth_proxy"])

    def test_compare_to_market_ev_enriches_confidence_with_missing_observed_ev_flag(self):
        implied = pd.DataFrame([{
            "Ticker": "TEST",
            "EV_Implied": 1000.0,
            "Avg_FCF_Used": 100.0,
            "Growth_Used": 0.03,
            "WACC_Used": 0.10,
            "Years": 5,
            "Assumptions_Used": {"as_of_date": "2026-02-22"},
            "Valuation_Confidence": vit._valuation_confidence_from_flags({}, context="dcf_implied_enterprise_value"),
            "Notes": "",
        }])

        with patch.object(valuation, "dcf_implied_enterprise_value", return_value=implied):
            with patch.object(valuation.yf, "Ticker", return_value=_FakeTicker({"enterpriseValue": None, "marketCap": None})):
                out = vit.compare_to_market_ev("TEST", as_df=True, analysis_report_date="2026-02-22")

        conf = out.loc[0, "Valuation_Confidence"]
        self.assertEqual(conf["context"], "compare_to_market_ev")
        self.assertTrue(conf["flags"]["missing_observed_ev"])

    def test_compare_to_market_cap_enriches_confidence_flags(self):
        ev_df = pd.DataFrame([{
            "Ticker": "TEST",
            "EV_Implied": 1000.0,
            "Avg_FCF_Used": 100.0,
            "Growth_Used": 0.03,
            "WACC_Used": 0.10,
            "Years": 5,
            "Assumptions_Used": {"as_of_date": "2026-02-22"},
            "Valuation_Confidence": vit._valuation_confidence_from_flags({}, context="compare_to_market_ev"),
            "Notes": "",
        }])
        eq_df = pd.DataFrame([{
            "Ticker": "TEST",
            "Equity_Implied": 800.0,
            "TotalDebt": 300.0,
            "CashAndCashEquivalents": 100.0,
            "MinorityInterest": 0.0,
            "SharesOutstanding": None,
            "Notes": "",
        }])

        with patch.object(valuation, "implied_equity_value_from_ev", return_value=eq_df):
            with patch.object(valuation.yf, "Ticker", return_value=_FakeTicker({"marketCap": None, "sharesOutstanding": None, "currentPrice": None})):
                out = vit.compare_to_market_cap(ev_df, as_df=True, analysis_report_date="2026-02-22")

        conf = out.loc[0, "Valuation_Confidence"]
        self.assertEqual(conf["context"], "compare_to_market_cap")
        self.assertTrue(conf["flags"]["missing_observed_market_cap"])
        self.assertTrue(conf["flags"]["missing_shares_for_per_share"])


if __name__ == "__main__":
    unittest.main()

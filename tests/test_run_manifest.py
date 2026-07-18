import unittest
from unittest.mock import patch

import pandas as pd

import ValueInvestingTools as vit
import vitlib.orchestrator as orchestrator
import vitlib.valuation as valuation


class TestRunManifest(unittest.TestCase):
    def test_valuation_defaults_do_not_mislabel_default_macro_values(self):
        payload = vit.valuation_defaults(
            as_of_date="2026-07-18",
            risk_free_rate=vit.VALUATION_DEFAULTS["risk_free_rate"],
            equity_risk_premium=vit.VALUATION_DEFAULTS["equity_risk_premium"],
        )

        self.assertEqual(payload["macro_inputs"]["risk_free_rate"]["source"], "macro_config")
        self.assertEqual(payload["macro_inputs"]["equity_risk_premium"]["source"], "macro_config")

    def test_build_run_manifest_records_inputs_outputs_and_assumptions(self):
        assumptions = vit.valuation_defaults(as_of_date="2026-07-18")
        manifest = vit.build_run_manifest(
            ticker="msft",
            analysis_report_date="2026-07-18",
            assumptions_used=assumptions,
            inputs={"function": "unit_test", "years": 5},
            outputs={"EV_Implied": 123.4},
            data_as_of={"price_asof": "2026-07-17"},
            health_notes=["ok"],
        )

        self.assertEqual(manifest["ticker"], "MSFT")
        self.assertTrue(manifest["manifest_id"].startswith("vit-run-"))
        self.assertEqual(manifest["inputs"]["years"], 5)
        self.assertEqual(manifest["outputs"]["EV_Implied"], 123.4)
        self.assertIn("risk_free_rate", manifest["assumptions"])
        self.assertEqual(manifest["health"], ["ok"])

    def test_dcf_implied_ev_includes_run_manifest(self):
        snap = {
            "ticker": "TEST",
            "beta": 1.0,
            "cashflow": pd.DataFrame(),
            "financials": pd.DataFrame(),
            "balance": pd.DataFrame(),
            "revenue_series": pd.Series([100.0, 110.0, 120.0]),
            "market_cap": 1000.0,
        }
        fcf_series = pd.Series([50.0, 55.0, 60.0, 65.0], index=[0, 1, 2, 3])

        with patch.object(valuation, "_pull_company_snapshot", return_value=snap):
            with patch.object(valuation, "_calculate_wacc", return_value=0.10):
                with patch.object(valuation, "_fcf_series_from_cashflow", return_value=fcf_series):
                    with patch.object(valuation, "_normalized_fcf_baseline", return_value=60.0):
                        with patch.object(valuation, "_fcf_cagr_details", return_value=(0.04, None, 15.0)):
                            out = vit.dcf_implied_enterprise_value(
                                "TEST",
                                years=5,
                                risk_free_rate=0.05,
                                as_df=True,
                                analysis_report_date="2026-07-18",
                            )

        self.assertIn("Run_Manifest", out.columns)
        manifest = out.loc[0, "Run_Manifest"]
        self.assertEqual(manifest["source"], "dcf_implied_enterprise_value")
        self.assertEqual(manifest["outputs"]["EV_Implied"], out.loc[0, "EV_Implied"])
        self.assertEqual(manifest["assumptions"]["risk_free_rate"]["source"], "user_override")

    def test_dcf_three_scenarios_includes_run_manifest(self):
        snap = {
            "ticker": "TEST",
            "beta": 1.0,
            "cashflow": pd.DataFrame(),
            "financials": pd.DataFrame(),
            "balance": pd.DataFrame(),
            "revenue_series": pd.Series([100.0, 110.0, 120.0]),
            "market_cap": 1000.0,
            "shares_outstanding": 100.0,
        }
        fcf_series = pd.Series([50.0, 55.0, 60.0, 65.0], index=[0, 1, 2, 3])

        with patch.object(valuation, "_pull_company_snapshot", return_value=snap):
            with patch.object(valuation, "_calculate_wacc", return_value=0.10):
                with patch.object(valuation, "_fcf_series_from_cashflow", return_value=fcf_series):
                    with patch.object(valuation, "_normalized_fcf_baseline", return_value=60.0):
                        with patch.object(valuation, "_fcf_cagr_details", return_value=(0.04, None, 15.0)):
                            out = vit.dcf_three_scenarios(
                                "TEST",
                                years=5,
                                as_df=True,
                                analysis_report_date="2026-07-18",
                            )

        self.assertIn("Run_Manifest", out.columns)
        manifest = out.loc[0, "Run_Manifest"]
        self.assertEqual(manifest["source"], "dcf_three_scenarios")
        self.assertEqual(manifest["outputs"]["Scenario"], out.loc[0, "Scenario"])

    def test_orchestrator_returns_consolidated_run_manifest(self):
        assumptions = vit.valuation_defaults(as_of_date="2026-07-18")
        component_manifest = vit.build_run_manifest(
            ticker="TEST",
            analysis_report_date="2026-07-18",
            assumptions_used=assumptions,
            inputs={"function": "component"},
            outputs={"EV_Implied": 1000.0},
        )
        prices = pd.DataFrame([{
            "Ticker": "TEST",
            "avg_price_1d": 10.0,
            "avg_price_30d": 11.0,
            "avg_price_90d": 12.0,
            "avg_price_180d": 13.0,
            "price_asof": "2026-07-17",
            "Notes": "",
        }])
        simple = pd.DataFrame([{"Ticker": "TEST", "Notes": ""}])
        ev = pd.DataFrame([{
            "Ticker": "TEST",
            "EV_Implied": 1000.0,
            "Assumptions_Used": assumptions,
            "Run_Manifest": component_manifest,
            "Notes": "",
        }])
        cap = pd.DataFrame([{
            "Ticker": "TEST",
            "Equity_Implied": 900.0,
            "Assumptions_Used": assumptions,
            "Run_Manifest": component_manifest,
            "Notes": "",
        }])
        dcf = pd.DataFrame([{
            "Scenario": "DCF_Mid_Growth_Mid_WACC",
            "Per_Share_Value": 20.0,
            "Assumptions_Used": assumptions,
            "Run_Manifest": component_manifest,
            "Notes": "",
        }])

        with patch.object(orchestrator, "historical_average_share_prices", return_value=prices):
            with patch.object(orchestrator, "historical_growth_metrics", return_value=simple):
                with patch.object(orchestrator, "compute_fundamentals_actuals", return_value=simple):
                    with patch.object(orchestrator, "compute_fundamentals_scores", return_value=simple):
                        with patch.object(orchestrator, "peer_multiples", return_value={"notes": []}):
                            with patch.object(orchestrator, "price_from_peer_multiples", return_value=simple):
                                with patch.object(orchestrator, "compare_to_market_ev", return_value=ev):
                                    with patch.object(orchestrator, "compare_to_market_cap", return_value=cap):
                                        with patch.object(orchestrator, "dcf_three_scenarios", return_value=dcf):
                                            out = orchestrator.orchestrator_function(
                                                "TEST",
                                                ["AAA", "BBB"],
                                                save_csv=False,
                                                analysis_report_date="2026-07-18",
                                            )

        self.assertIn("run_manifest", out)
        manifest = out["run_manifest"]
        self.assertEqual(manifest["source"], "orchestrator_function")
        self.assertIn(component_manifest["manifest_id"], manifest["outputs"]["component_manifest_ids"])


if __name__ == "__main__":
    unittest.main()

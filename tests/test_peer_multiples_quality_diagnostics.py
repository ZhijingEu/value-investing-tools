import unittest
from unittest.mock import patch

import ValueInvestingTools as vit
import vitlib.peers as peers


class TestPeerMultiplesQualityDiagnostics(unittest.TestCase):
    def test_peer_quality_diagnostics_emits_coverage_and_dispersion_warnings(self):
        snapshots = {
            "AAA": {
                "ticker": "AAA",
                "pe_ratio": 22.0, "trailing_pe_ratio": 22.0, "forward_pe_ratio": 20.0,
                "ps_ratio": 4.0, "ev_to_ebitda": 12.0,
                "market_cap": 500_000_000_000,
                "revenue": 100_000_000_000,
                "revenue_growth": 0.12,
                "earnings_growth": 0.18,
                "operating_margin": 0.25,
                "debt_to_equity_info": 45.0,
                "beta": 1.1,
                "ebitda": 35_000_000_000,
            },
            "BBB": {
                "ticker": "BBB",
                "pe_ratio": 18.0, "trailing_pe_ratio": 18.0, "forward_pe_ratio": 16.0,
                "ps_ratio": 2.0, "ev_to_ebitda": 8.0,
                "market_cap": 20_000_000_000,
                "revenue": 5_000_000_000,
                "revenue_growth": None,  # low coverage trigger on peers once target is excluded
                "earnings_growth": -0.05,
                "operating_margin": 0.08,
                "debt_to_equity_info": 220.0,
                "beta": 1.8,
                "ebitda": 400_000_000,
            },
            "CCC": {
                "ticker": "CCC",
                "pe_ratio": 14.0, "trailing_pe_ratio": 14.0, "forward_pe_ratio": None,
                "ps_ratio": 1.5, "ev_to_ebitda": 7.0,
                "market_cap": 2_000_000_000,
                "revenue": 800_000_000,
                "revenue_growth": 0.01,
                "earnings_growth": None,
                "operating_margin": 0.03,
                "debt_to_equity_info": 15.0,
                "beta": 0.8,
                "ebitda": 40_000_000,
            },
        }

        with patch.object(peers, "_pull_company_snapshot", side_effect=lambda t: snapshots[t]):
            out = vit.peer_multiples(
                ["AAA", "BBB", "CCC"],
                target_ticker="AAA",
                include_target=False,
                as_df=True,
            )

        diag = out["peer_quality_diagnostics"]
        self.assertEqual(diag["peer_count_for_stats"], 2)  # AAA excluded from peer stats
        self.assertIn("metrics", diag)
        self.assertIn("warnings", diag)

        metrics = {m["metric"]: m for m in diag["metrics"]}
        # revenue_growth among peers BBB/CCC has only one non-null value -> low-coverage warning
        self.assertEqual(metrics["revenue_growth"]["status"], "warn_low_coverage")
        self.assertTrue(any("Revenue Growth" in w for w in diag["warnings"]))
        # leverage dispersion should warn because BBB vs CCC debt/equity differs materially
        self.assertTrue(str(metrics["debt_to_equity_info"]["status"]).startswith("warn"))
        self.assertTrue(any("Debt/Equity" in w for w in diag["warnings"]))


if __name__ == "__main__":
    unittest.main()

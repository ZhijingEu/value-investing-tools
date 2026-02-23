import unittest

import ValueInvestingTools as vit


class TestValuationDefaults(unittest.TestCase):
    def test_defaults_payload_uses_expected_base_values(self):
        payload = vit.valuation_defaults()
        self.assertIn("as_of_date", payload)
        self.assertEqual(payload["risk_free_rate"], 0.0418)
        self.assertEqual(payload["equity_risk_premium"], 0.0423)
        self.assertEqual(payload["target_cagr_fallback"], 0.020)
        self.assertEqual(payload["fcf_window_years"], 3)
        self.assertEqual(payload["terminal_growth_gap"], 0.005)
        self.assertIn("assumptions_schema_version", payload)
        self.assertIn("assumptions_source", payload)
        self.assertIn("assumptions_snapshot_id", payload)
        self.assertTrue(str(payload["assumptions_snapshot_id"]).startswith("vit-val-"))

    def test_defaults_payload_accepts_overrides(self):
        payload = vit.valuation_defaults(
            as_of_date="2026-02-20",
            risk_free_rate=0.0449,
            equity_risk_premium=0.0417,
            target_cagr_fallback=0.018,
            fcf_window_years=2,
        )
        self.assertEqual(payload["as_of_date"], "2026-02-20")
        self.assertEqual(payload["risk_free_rate"], 0.0449)
        self.assertEqual(payload["equity_risk_premium"], 0.0417)
        self.assertEqual(payload["target_cagr_fallback"], 0.018)
        self.assertEqual(payload["fcf_window_years"], 2)
        self.assertEqual(payload["assumptions_schema_version"], "1.0")

    def test_assumptions_snapshot_id_is_stable_and_changes_on_input_change(self):
        p1 = vit.valuation_defaults(
            as_of_date="2026-02-22",
            risk_free_rate=0.0418,
            equity_risk_premium=0.0423,
            target_cagr_fallback=0.020,
            fcf_window_years=3,
        )
        p2 = vit.valuation_defaults(
            as_of_date="2026-02-22",
            risk_free_rate=0.0418,
            equity_risk_premium=0.0423,
            target_cagr_fallback=0.020,
            fcf_window_years=3,
        )
        p3 = vit.valuation_defaults(
            as_of_date="2026-02-22",
            risk_free_rate=0.050,  # changed
            equity_risk_premium=0.0423,
            target_cagr_fallback=0.020,
            fcf_window_years=3,
        )
        self.assertEqual(p1["assumptions_snapshot_id"], p2["assumptions_snapshot_id"])
        self.assertNotEqual(p1["assumptions_snapshot_id"], p3["assumptions_snapshot_id"])


if __name__ == "__main__":
    unittest.main()

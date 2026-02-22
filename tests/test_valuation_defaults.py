import unittest

import ValueInvestingTools as vit


class TestValuationDefaults(unittest.TestCase):
    def test_defaults_payload_uses_expected_base_values(self):
        payload = vit.valuation_defaults()
        self.assertIn("as_of_date", payload)
        self.assertEqual(payload["risk_free_rate"], 0.045)
        self.assertEqual(payload["equity_risk_premium"], 0.060)
        self.assertEqual(payload["target_cagr_fallback"], 0.020)
        self.assertEqual(payload["fcf_window_years"], 3)
        self.assertEqual(payload["terminal_growth_gap"], 0.005)

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


if __name__ == "__main__":
    unittest.main()

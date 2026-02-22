import unittest
from unittest.mock import patch

import ValueInvestingTools as vit


class TestPeerMultiplesForwardMode(unittest.TestCase):
    def test_forward_pe_mode_uses_forward_pe_with_trailing_fallback(self):
        snapshots = {
            "AAA": {
                "ticker": "AAA",
                "pe_ratio": 30.0,
                "trailing_pe_ratio": 30.0,
                "forward_pe_ratio": 24.0,
                "ps_ratio": 5.0,
                "ev_to_ebitda": 15.0,
            },
            "BBB": {
                "ticker": "BBB",
                "pe_ratio": 25.0,
                "trailing_pe_ratio": 25.0,
                "forward_pe_ratio": 20.0,
                "ps_ratio": 4.0,
                "ev_to_ebitda": 12.0,
            },
            "CCC": {
                "ticker": "CCC",
                "pe_ratio": 15.0,
                "trailing_pe_ratio": 15.0,
                "forward_pe_ratio": None,
                "ps_ratio": 3.0,
                "ev_to_ebitda": 9.0,
            },
        }

        def _fake_pull(symbol):
            return snapshots[symbol]

        with patch.object(vit, "_pull_company_snapshot", side_effect=_fake_pull):
            out = vit.peer_multiples(
                ["AAA", "BBB", "CCC"],
                target_ticker="AAA",
                include_target=False,
                multiple_basis="forward_pe",
                as_df=True,
            )

        detail = out["peer_comp_detail"]
        peer_bands = out["peer_multiple_bands_wide"]

        # Target row is preserved in detail and uses forward PE in this mode.
        aaa = detail.loc[detail["ticker"] == "AAA"].iloc[0]
        self.assertEqual(float(aaa["pe_ratio"]), 24.0)
        self.assertEqual(aaa["pe_ratio_basis"], "forwardPE")

        # Peer rows (BBB, CCC) should use forward PE when available, else trailing fallback.
        bbb = detail.loc[detail["ticker"] == "BBB"].iloc[0]
        ccc = detail.loc[detail["ticker"] == "CCC"].iloc[0]
        self.assertEqual(float(bbb["pe_ratio"]), 20.0)
        self.assertEqual(bbb["pe_ratio_basis"], "forwardPE")
        self.assertEqual(float(ccc["pe_ratio"]), 15.0)
        self.assertEqual(ccc["pe_ratio_basis"], "trailingPE_fallback")

        # Peer bands exclude target AAA and should therefore reflect [20, 15] for PE.
        self.assertAlmostEqual(float(peer_bands.loc["PE", "Min"]), 15.0)
        self.assertAlmostEqual(float(peer_bands.loc["PE", "Max"]), 20.0)

        # Metadata should explicitly state mixed-basis behavior.
        self.assertEqual(out["multiple_basis"], "forward_pe")
        self.assertIn("fallback", out["metric_basis_map"]["PE"])
        self.assertTrue(any("PS and EV/EBITDA remain trailing" in n for n in out["notes"]))


if __name__ == "__main__":
    unittest.main()

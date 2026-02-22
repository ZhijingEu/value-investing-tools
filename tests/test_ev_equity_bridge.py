import unittest

import ValueInvestingTools as vit


class TestEvEquityBridge(unittest.TestCase):
    def test_equity_from_ev_with_net_debt_does_not_double_count_cash(self):
        # EV - NetDebt - MinorityInterest
        equity = vit._equity_value_from_ev(
            1000.0,
            total_debt=300.0,
            cash_eq=100.0,
            minority_interest=50.0,
            net_debt=200.0,
        )
        self.assertEqual(equity, 750.0)

    def test_equity_from_ev_falls_back_to_debt_cash_formula(self):
        # EV - TotalDebt + Cash - MinorityInterest
        equity = vit._equity_value_from_ev(
            1000.0,
            total_debt=300.0,
            cash_eq=100.0,
            minority_interest=50.0,
            net_debt=None,
        )
        self.assertEqual(equity, 750.0)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from unittest.mock import patch
from pathlib import Path

DEPENDENCY_ERROR = None
try:
    import pandas as pd
    import server
except ModuleNotFoundError as exc:  # pragma: no cover - local env dependency gap
    pd = None
    server = None
    DEPENDENCY_ERROR = exc


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"MCP contract tests skipped (missing dependency): {DEPENDENCY_ERROR}")
class TestMcpResponseContracts(unittest.TestCase):
    def test_text_item_contract(self):
        item = server.text_item("hello")
        self.assertEqual(item["type"], "text")
        self.assertEqual(item["text"], "hello")

    def test_file_resource_preserves_file_uri(self):
        uri = "file:///C:/tmp/example.csv"
        item = server.file_resource(uri, "text/csv")
        self.assertEqual(item["type"], "resource")
        self.assertEqual(item["uri"], uri)
        self.assertEqual(item["mimeType"], "text/csv")

    def test_tool_health_to_markdown_returns_single_text_item(self):
        with patch.object(server.vit, "health_to_markdown", return_value="# Health\nOK"):
            out = server.tool_health_to_markdown([{"source": "x"}])

        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["type"], "text")
        self.assertIn("OK", out[0]["text"])

    def test_tool_health_to_tables_returns_json_text_records(self):
        df = pd.DataFrame([{"source": "dcf", "ticker": "MSFT", "notes": "ok"}])
        with patch.object(server.vit, "health_to_tables", return_value=df):
            out = server.tool_health_to_tables([{"source": "dcf"}], sort=True)

        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["type"], "text")
        rows = json.loads(out[0]["text"])
        self.assertIsInstance(rows, list)
        self.assertEqual(rows[0]["source"], "dcf")
        self.assertEqual(rows[0]["ticker"], "MSFT")

    def test_tool_dcf_sensitivity_grid_returns_json_payload(self):
        grid_long = pd.DataFrame(
            [{"Growth_Input": 0.02, "WACC_Input": 0.1, "Per_Share_Value": 12.3}]
        )
        grid_wide = pd.DataFrame({0.1: [12.3]}, index=[0.02])
        fake = {
            "ticker": "MSFT",
            "analysis_report_date": "2026-02-22",
            "grid_long": grid_long,
            "grid_wide": grid_wide,
            "inputs_used": {"years": 5},
            "notes": ["ok"],
        }
        with patch.object(server.vit, "dcf_sensitivity_grid", return_value=fake):
            out = server.tool_dcf_sensitivity_grid("MSFT")

        self.assertIsInstance(out, list)
        self.assertGreaterEqual(len(out), 2)
        self.assertEqual(out[0]["type"], "text")
        self.assertEqual(out[1]["type"], "text")
        payload = json.loads(out[1]["text"])
        self.assertEqual(payload["ticker"], "MSFT")
        self.assertIn("grid_wide", payload)
        self.assertIn("grid_long", payload)
        self.assertIn("inputs_used", payload)

    def test_tool_peer_multiples_returns_expected_payload(self):
        peer_comp_detail = pd.DataFrame([{"ticker": "AAA", "pe_ratio": 10.0}])
        peer_multiple_bands_wide = pd.DataFrame({"Min": [10.0]}, index=["PE"])
        peer_comp_bands = pd.DataFrame([{"Scenario": "PE_Min", "Valuation_per_Share": 123.0}])
        fake = {
            "target_ticker": "AAA",
            "multiple_basis": "forward_pe",
            "metric_basis_map": {"PE": "forwardPE"},
            "notes": ["ok"],
            "peer_quality_diagnostics": {"peer_count_for_stats": 1, "metrics": [], "warnings": []},
            "peer_comp_detail": peer_comp_detail,
            "peer_multiple_bands_wide": peer_multiple_bands_wide,
            "peer_comp_bands": peer_comp_bands,
        }
        with patch.object(server.vit, "peer_multiples", return_value=fake):
            out = server.tool_peer_multiples(["AAA", "BBB"], target_ticker="AAA", include_target=False, multiple_basis="forward_pe")

        self.assertIsInstance(out, list)
        self.assertGreaterEqual(len(out), 4)
        payload = json.loads(out[-1]["text"])
        self.assertEqual(payload["target_ticker"], "AAA")
        self.assertEqual(payload["multiple_basis"], "forward_pe")
        self.assertIn("metric_basis_map", payload)
        self.assertIn("peer_quality_diagnostics", payload)

    def test_tool_compare_to_market_ev_returns_confidence(self):
        df = pd.DataFrame([{
            "Ticker": "TEST",
            "Observed_EV": 1000.0,
            "EV_Implied": 900.0,
            "Premium_%": 11.1,
            "Valuation_Confidence": {"score": 0.8, "level": "high"},
            "Assumptions_Used": {"assumptions_snapshot_id": "vit-val-123"},
        }])
        with patch.object(server.vit, "compare_to_market_ev", return_value=df):
            out = server.tool_compare_to_market_ev("TEST")

        self.assertIsInstance(out, list)
        payload = json.loads(out[0]["text"])
        self.assertEqual(payload[0]["Valuation_Confidence"]["level"], "high")
        self.assertEqual(payload[0]["Assumptions_Used"]["assumptions_snapshot_id"], "vit-val-123")

    def test_plot_scores_clustered_returns_image_resource_contract(self):
        class _DummyFig:
            def savefig(self, *_args, **_kwargs):
                return None

        with patch.object(server.vit, "plot_scores_clustered", return_value=(_DummyFig(), object())):
            with patch.object(
                server,
                "image_items_for_png",
                return_value=[{"type": "resource", "mimeType": "image/png", "uri": "file:///tmp/chart.png"}],
            ):
                out = server.tool_plot_scores_clustered(["MSFT", "AAPL"])

        self.assertIsInstance(out, list)
        self.assertEqual(out[0]["type"], "resource")
        self.assertEqual(out[0]["mimeType"], "image/png")
        self.assertTrue(str(out[0]["uri"]).startswith("file://"))

    def test_plot_single_metric_ts_returns_image_resource_contract(self):
        class _DummyFig:
            def savefig(self, *_args, **_kwargs):
                return None

        local_tmp_dir = Path(".tmp") / "mcp_contracts"
        local_tmp_dir.mkdir(parents=True, exist_ok=True)
        with patch.object(server.vit, "plot_single_metric_ts", return_value=(_DummyFig(), object())):
            with patch.object(server, "ticker_dir", return_value=local_tmp_dir):
                with patch.object(
                    server,
                    "image_items_for_png",
                    return_value=[{"type": "resource", "mimeType": "image/png", "uri": "file:///tmp/metric.png"}],
                ):
                    out = server.tool_plot_single_metric_ts("MSFT", "ROE")

        self.assertIsInstance(out, list)
        self.assertEqual(out[0]["type"], "resource")
        self.assertEqual(out[0]["mimeType"], "image/png")


if __name__ == "__main__":
    unittest.main()

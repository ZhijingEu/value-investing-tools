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

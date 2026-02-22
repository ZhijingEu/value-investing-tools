from __future__ import annotations

from typing import Any, Mapping

import yfinance as yf

from .base import MarketDataProvider, ProviderMeta


class YahooFinanceProvider(MarketDataProvider):
    """
    yfinance-backed provider adapter.

    Sprint 1 scope:
    - light wrapper used for future dependency inversion
    - not yet wired into ValueInvestingTools.py call paths
    """

    @property
    def meta(self) -> ProviderMeta:
        version = getattr(yf, "__version__", None)
        return ProviderMeta(name="yfinance", version=version)

    def ticker(self, symbol: str) -> Any:
        return yf.Ticker(symbol)

    def info(self, symbol: str) -> Mapping[str, Any]:
        return self.ticker(symbol).info

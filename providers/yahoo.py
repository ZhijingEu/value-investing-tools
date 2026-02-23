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

    def __init__(self) -> None:
        self._ticker_cache: dict[str, Any] = {}
        self._data_cache: dict[tuple, Any] = {}

    @property
    def meta(self) -> ProviderMeta:
        version = getattr(yf, "__version__", None)
        return ProviderMeta(name="yfinance", version=version)

    def ticker(self, symbol: str) -> Any:
        key = symbol.replace(".", "-").strip().upper()
        if key not in self._ticker_cache:
            self._ticker_cache[key] = yf.Ticker(key)
        return self._ticker_cache[key]

    def info(self, symbol: str) -> Mapping[str, Any]:
        key = (symbol.replace(".", "-").strip().upper(), "info")
        if key in self._data_cache:
            return self._data_cache[key]
        data = self.ticker(symbol).info
        self._data_cache[key] = data
        return data

    def financials(self, symbol: str) -> Any:
        key = (symbol.replace(".", "-").strip().upper(), "financials")
        if key in self._data_cache:
            return self._data_cache[key]
        data = self.ticker(symbol).financials
        self._data_cache[key] = data
        return data

    def cashflow(self, symbol: str) -> Any:
        key = (symbol.replace(".", "-").strip().upper(), "cashflow")
        if key in self._data_cache:
            return self._data_cache[key]
        data = self.ticker(symbol).cashflow
        self._data_cache[key] = data
        return data

    def balance_sheet(self, symbol: str) -> Any:
        key = (symbol.replace(".", "-").strip().upper(), "balance_sheet")
        if key in self._data_cache:
            return self._data_cache[key]
        data = self.ticker(symbol).balance_sheet
        self._data_cache[key] = data
        return data

    def history(self, symbol: str, **kwargs: Any) -> Any:
        key = (symbol.replace(".", "-").strip().upper(), "history", tuple(sorted(kwargs.items())))
        if key in self._data_cache:
            return self._data_cache[key]
        data = self.ticker(symbol).history(**kwargs)
        self._data_cache[key] = data
        return data

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ProviderMeta:
    """Minimal provider metadata for auditability and debugging."""

    name: str
    version: str | None = None


class MarketDataProvider(Protocol):
    """
    Narrow interface for upstream market/fundamentals data providers.

    This is intentionally small in Sprint 1:
    - it establishes a stable seam for future refactors
    - it does not force immediate migration of existing yfinance-based code
    """

    @property
    def meta(self) -> ProviderMeta:
        ...

    def ticker(self, symbol: str) -> Any:
        """Return a provider-native ticker/client object."""
        ...

    def info(self, symbol: str) -> Mapping[str, Any]:
        """Return snapshot metadata/quote fields for a symbol."""
        ...

    def financials(self, symbol: str) -> Any:
        """Return annual income statement data for a symbol."""
        ...

    def cashflow(self, symbol: str) -> Any:
        """Return annual cashflow statement data for a symbol."""
        ...

    def balance_sheet(self, symbol: str) -> Any:
        """Return annual balance sheet data for a symbol."""
        ...

    def history(self, symbol: str, **kwargs: Any) -> Any:
        """Return historical price data for a symbol."""
        ...

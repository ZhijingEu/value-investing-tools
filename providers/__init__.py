"""
Provider adapters for data-source access.

Sprint 1 scope: scaffold only (no runtime integration yet).
The valuation/fundamentals engines still read directly from yfinance today.
"""

from .base import MarketDataProvider
from .yahoo import YahooFinanceProvider

__all__ = ["MarketDataProvider", "YahooFinanceProvider"]

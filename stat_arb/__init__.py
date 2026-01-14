"""
Institutional-Quality Crypto Stat-Arb Backtest System
=====================================================

A modular statistical arbitrage backtesting framework for low-cap crypto
tokens with signed graph clustering, PCA market mode removal, and
rolling walk-forward out-of-sample testing.

Key Features:
- No lookahead bias: signals from t-1, trades at t close
- Signed k-NN graph construction with SPONGE/BNC/Spectral clustering
- PCA-based market mode removal and residualization
- Multiple signal strategies: z-score, cluster deviation, pairs trading
- Risk controls: dollar/PC1/ETH neutral, position caps, turnover limits
- Transaction cost modeling with sensitivity analysis
"""

__version__ = "1.0.0"

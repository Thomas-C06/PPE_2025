"""Backtesting engine for GeoQuant AI -- Bloc 3.

Simulates two strategies on historical data and computes performance metrics:
    - Buy & Hold : always invested (baseline)
    - GeoQuant   : follows the signal column (1=Long, 0=Cash)

Metrics computed:
    - Total return
    - Annualised return
    - Max Drawdown
    - Sharpe Ratio (annualised, risk-free rate = 0)
    - Win Rate (% of invested days with positive return)
    - Number of trades (signal transitions 0->1 or 1->0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Container for backtest results of a single strategy."""

    name:            str
    total_return:    float          # e.g. 0.35  = +35 %
    annualised:      float          # annualised CAGR
    max_drawdown:    float          # e.g. -0.20 = -20 %
    sharpe:          float
    win_rate:        float          # fraction of invested days with positive return
    nb_trades:       int
    equity_curve:    pd.Series      # cumulative value starting at 1.0
    daily_returns:   pd.Series
    trade_log:       pd.DataFrame   # date | action | price | pnl_pct

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Rendement total":    f"{self.total_return:+.2%}",
            "Rend. annualise":    f"{self.annualised:+.2%}",
            "Max Drawdown":       f"{self.max_drawdown:.2%}",
            "Sharpe Ratio":       f"{self.sharpe:.2f}",
            "Win Rate":           f"{self.win_rate:.1%}",
            "Nb trades":          str(self.nb_trades),
        }


def _equity_curve(daily_rets: pd.Series) -> pd.Series:
    """Return cumulative product starting at 1.0."""
    return (1 + daily_rets).cumprod()


def _max_drawdown(equity: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve."""
    running_max = equity.cummax()
    drawdown    = (equity - running_max) / running_max
    return float(drawdown.min())


def _sharpe(daily_rets: pd.Series) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0)."""
    std = daily_rets.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(daily_rets.mean() / std * np.sqrt(252))


def _annualised_return(total_ret: float, n_days: int) -> float:
    """CAGR from total return and number of calendar days."""
    if n_days <= 0:
        return 0.0
    years = n_days / 365.25
    return float((1 + total_ret) ** (1 / years) - 1)


def _build_trade_log(
    df: pd.DataFrame,
    strategy_rets: pd.Series,
    signal_col: str = "signal",
    price_col: str  = "Close",
) -> pd.DataFrame:
    """
    Build a trade log from signal transitions.

    Args:
        df:            Full dataset with 'date', signal_col and price_col.
        strategy_rets: Daily returns of the strategy.
        signal_col:    Column name for the signal (1=Long, 0=Cash).
        price_col:     Column name for the price.

    Returns:
        DataFrame with columns: date, action, price, pnl_pct.
    """
    records = []
    prev_sig = 0

    for i, row in df.iterrows():
        sig = int(row[signal_col]) if signal_col in df.columns else 1
        if sig != prev_sig:
            action = "ENTREE" if sig == 1 else "SORTIE"
            pnl    = float(strategy_rets.iloc[i]) if action == "SORTIE" else float("nan")
            records.append({
                "Date":   row["date"],
                "Action": action,
                "Prix":   round(float(row[price_col]), 2) if price_col in df.columns else float("nan"),
                "P&L %":  f"{pnl:+.2%}" if not np.isnan(pnl) else "--",
            })
            prev_sig = sig

    return pd.DataFrame(records)


def run_backtest(
    df: pd.DataFrame,
    price_col: str = "Close",
) -> tuple[BacktestResult, BacktestResult]:
    """
    Run Buy & Hold and GeoQuant backtests on the dataset.

    Args:
        df:        DataFrame with columns: date, Returns, signal, (price_col).
        price_col: Name of the price column for the trade log.

    Returns:
        Tuple (buy_hold_result, geoquant_result).
    """
    df = df.dropna(subset=["Returns"]).copy().reset_index(drop=True)

    n_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days

    # ── Buy & Hold ────────────────────────────────────────────────────────────
    bh_rets   = df["Returns"].fillna(0.0)
    bh_equity = _equity_curve(bh_rets)
    bh_total  = float(bh_equity.iloc[-1] - 1)
    bh_wr     = float((bh_rets[bh_rets != 0] > 0).mean())

    bh = BacktestResult(
        name          = "Buy & Hold",
        total_return  = bh_total,
        annualised    = _annualised_return(bh_total, n_days),
        max_drawdown  = _max_drawdown(bh_equity),
        sharpe        = _sharpe(bh_rets),
        win_rate      = bh_wr,
        nb_trades     = 1,
        equity_curve  = bh_equity,
        daily_returns = bh_rets,
        trade_log     = pd.DataFrame({"Date": [df["date"].iloc[0]], "Action": ["ENTREE"],
                                      "Prix": [round(float(df[price_col].iloc[0]), 2)
                                               if price_col in df.columns else float("nan")],
                                      "P&L %": ["--"]}),
    )

    # ── GeoQuant ──────────────────────────────────────────────────────────────
    # Le signal du jour J est connu en fin de journee J, applique le jour J+1
    signal_shifted = df["signal"].shift(1).fillna(0).astype(int)
    gq_rets        = df["Returns"].fillna(0.0) * signal_shifted
    gq_equity      = _equity_curve(gq_rets)
    gq_total       = float(gq_equity.iloc[-1] - 1)

    invested_days  = signal_shifted == 1
    gq_wr          = float((gq_rets[invested_days] > 0).mean()) if invested_days.any() else 0.0

    signal_changes = (df["signal"].diff().abs() > 0).sum()
    nb_trades      = int(signal_changes // 2) if signal_changes > 0 else 0

    gq = BacktestResult(
        name          = "GeoQuant",
        total_return  = gq_total,
        annualised    = _annualised_return(gq_total, n_days),
        max_drawdown  = _max_drawdown(gq_equity),
        sharpe        = _sharpe(gq_rets),
        win_rate      = gq_wr,
        nb_trades     = nb_trades,
        equity_curve  = gq_equity,
        daily_returns = gq_rets,
        trade_log     = _build_trade_log(df, gq_rets, "signal", price_col),
    )

    return bh, gq

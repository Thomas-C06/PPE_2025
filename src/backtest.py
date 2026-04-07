"""Backtesting engine for GeoQuant AI -- Bloc 3.

Simulates two strategies on historical data and computes performance metrics:
    - Buy & Hold : always invested at 100% (baseline)
    - GeoQuant   : follows 'position' column (float 0.0-1.0 from strategy.py)

Metrics computed:
    - Total return
    - Annualised CAGR
    - Max Drawdown
    - Sharpe Ratio (annualised, configurable risk-free rate)
    - Win Rate     (% of TRADES with positive P&L, not % of days)
    - Number of trades (round-trips: entry + exit = 1 trade)

Corrections vs v1:
    - Win rate is now computed per trade (entry -> exit), not per day
    - Transaction costs (costs_bps basis points per side) deducted on each trade
    - Sharpe computed on excess returns (daily_ret - risk_free_daily)
    - Position uses float sizing (0.0-1.0) not binary signal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Container for backtest results of a single strategy."""

    name:          str
    total_return:  float        # e.g. 0.35  = +35 %
    annualised:    float        # CAGR
    max_drawdown:  float        # e.g. -0.20 = -20 %
    sharpe:        float        # annualised Sharpe on excess returns
    win_rate:      float        # % of completed trades with positive P&L
    nb_trades:     int          # number of round-trip trades
    equity_curve:  pd.Series    # cumulative value starting at 1.0
    daily_returns: pd.Series    # net daily returns (after costs)
    trade_log:     pd.DataFrame # date | action | prix | pnl_pct | position

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Rendement total": f"{self.total_return:+.2%}",
            "Rend. annualise": f"{self.annualised:+.2%}",
            "Max Drawdown":    f"{self.max_drawdown:.2%}",
            "Sharpe Ratio":    f"{self.sharpe:.2f}",
            "Win Rate":        f"{self.win_rate:.1%}" if not np.isnan(self.win_rate) else "N/A",
            "Nb trades":       str(self.nb_trades),
        }


# ── Fonctions utilitaires ──────────────────────────────────────────────────────

def _equity_curve(daily_rets: pd.Series) -> pd.Series:
    return (1 + daily_rets).cumprod()


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return float(dd.min())


def _sharpe(daily_rets: pd.Series, risk_free_annual: float = 0.0) -> float:
    """
    Annualised Sharpe ratio on excess returns.

    Args:
        daily_rets:        Series of daily strategy returns.
        risk_free_annual:  Annual risk-free rate (e.g. 0.05 for 5%).
    """
    rf_daily = risk_free_annual / 252
    excess   = daily_rets - rf_daily
    std      = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))


def _annualised_return(total_ret: float, n_trading_days: int) -> float:
    """CAGR using trading days (more accurate than calendar days)."""
    if n_trading_days <= 0:
        return 0.0
    years = n_trading_days / 252
    return float((1 + total_ret) ** (1 / years) - 1)


def _apply_costs(
    daily_rets: pd.Series,
    position: pd.Series,
    costs_bps: float,
) -> pd.Series:
    """
    Subtract transaction costs on each position change.

    A cost of costs_bps basis points is applied per side (entry OR exit).
    The cost is deducted from the daily return on the day of the trade.

    Args:
        daily_rets: Series of raw daily returns.
        position:   Float position series (0.0 to 1.0).
        costs_bps:  Cost per side in basis points (10 bps = 0.10%).
    """
    cost_per_side = costs_bps / 10_000
    pos_change    = position.diff().abs().fillna(0.0)
    # cost = change_in_position * cost_per_side
    cost_series   = pos_change * cost_per_side
    return daily_rets - cost_series


def _win_rate_and_log(
    df: pd.DataFrame,
    net_rets: pd.Series,
    position: pd.Series,
    price_col: str,
) -> Tuple[float, pd.DataFrame, int]:
    """
    Compute per-trade win rate and build the trade log.

    A trade is defined as the period between a position increase (entry)
    and a return to 0 (exit). Win = exit value > entry value.

    Returns:
        (win_rate, trade_log_df, nb_trades)
    """
    records: List[dict] = []
    trades_won  = 0
    trades_total = 0

    entry_price: float | None = None
    entry_date              = None
    entry_pos               = 0.0

    pos_vals = position.values
    prices   = df[price_col].values if price_col in df.columns else np.full(len(df), np.nan)
    dates    = df["date"].values

    for i in range(len(df)):
        pos   = float(pos_vals[i])
        price = float(prices[i])
        date  = dates[i]

        prev_pos = float(pos_vals[i - 1]) if i > 0 else 0.0

        # ENTREE : position increases from 0 to >0
        if prev_pos == 0.0 and pos > 0.0:
            entry_price = price
            entry_date  = date
            entry_pos   = pos
            records.append({
                "Date":     date,
                "Action":   "ENTREE",
                "Prix":     round(price, 2),
                "Position": f"{pos:.0%}",
                "P&L %":    "--",
            })

        # SORTIE : position drops to 0
        elif prev_pos > 0.0 and pos == 0.0 and entry_price is not None:
            pnl = (price - entry_price) / entry_price
            records.append({
                "Date":     date,
                "Action":   "SORTIE",
                "Prix":     round(price, 2),
                "Position": "0%",
                "P&L %":    f"{pnl:+.2%}",
            })
            trades_total += 1
            if pnl > 0:
                trades_won += 1
            entry_price = None

    # Fermeture de la derniere position ouverte a la fin de la periode
    if entry_price is not None and len(prices) > 0:
        last_price = float(prices[-1])
        last_date  = dates[-1]
        pnl = (last_price - entry_price) / entry_price
        records.append({
            "Date":     last_date,
            "Action":   "SORTIE (fin periode)",
            "Prix":     round(last_price, 2),
            "Position": "0%",
            "P&L %":    f"{pnl:+.2%}",
        })
        trades_total += 1
        if pnl > 0:
            trades_won += 1

    win_rate = trades_won / trades_total if trades_total > 0 else float("nan")
    trade_df = pd.DataFrame(records)
    return win_rate, trade_df, trades_total


# ── Fonction principale ────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    price_col: str        = "Close",
    costs_bps: float      = 2.0,
    risk_free_annual: float = 0.0,
) -> Tuple[BacktestResult, BacktestResult]:
    """
    Run Buy & Hold and GeoQuant backtests.

    Args:
        df:               DataFrame with: date, Returns, signal, position, price_col.
        price_col:        Price column for trade log.
        costs_bps:        Transaction cost per side in basis points (default 2 bps, realistic for S&P 500 ETF).
        risk_free_annual: Annual risk-free rate for Sharpe (default 0%).

    Returns:
        Tuple (buy_hold_result, geoquant_result).
    """
    df = df.dropna(subset=["Returns"]).copy().reset_index(drop=True)

    n_trading_days = len(df)

    # ── Buy & Hold ─────────────────────────────────────────────────────────────
    bh_pos    = pd.Series(1.0, index=df.index)
    bh_raw    = df["Returns"].fillna(0.0)
    bh_rets   = _apply_costs(bh_raw, bh_pos, costs_bps)
    bh_equity = _equity_curve(bh_rets)
    bh_total  = float(bh_equity.iloc[-1] - 1)

    # Win rate B&H: % of positive days (conventional)
    bh_wr = float((bh_rets > 0).mean())

    bh = BacktestResult(
        name          = "Buy & Hold",
        total_return  = bh_total,
        annualised    = _annualised_return(bh_total, n_trading_days),
        max_drawdown  = _max_drawdown(bh_equity),
        sharpe        = _sharpe(bh_rets, risk_free_annual),
        win_rate      = bh_wr,
        nb_trades     = 1,
        equity_curve  = bh_equity,
        daily_returns = bh_rets,
        trade_log     = pd.DataFrame({
            "Date":     [df["date"].iloc[0]],
            "Action":   ["ENTREE"],
            "Prix":     [round(float(df[price_col].iloc[0]), 2)
                         if price_col in df.columns else float("nan")],
            "Position": ["100%"],
            "P&L %":    ["--"],
        }),
    )

    # ── GeoQuant ───────────────────────────────────────────────────────────────
    # Position du jour J est connue en fin de journee J, appliquee le jour J+1
    # Note: strategy.py a deja decale geo_score d'un jour; on decale ici la position
    # d'un jour supplementaire pour la realisme de l'execution.
    position_shifted = df["position"].shift(1).fillna(0.0)

    gq_raw    = df["Returns"].fillna(0.0) * position_shifted
    gq_rets   = _apply_costs(gq_raw, position_shifted, costs_bps)
    gq_equity = _equity_curve(gq_rets)
    gq_total  = float(gq_equity.iloc[-1] - 1)

    gq_wr, trade_log, nb_trades = _win_rate_and_log(
        df, gq_rets, position_shifted, price_col
    )

    gq = BacktestResult(
        name          = "GeoQuant",
        total_return  = gq_total,
        annualised    = _annualised_return(gq_total, n_trading_days),
        max_drawdown  = _max_drawdown(gq_equity),
        sharpe        = _sharpe(gq_rets, risk_free_annual),
        win_rate      = gq_wr,
        nb_trades     = nb_trades,
        equity_curve  = gq_equity,
        daily_returns = gq_rets,
        trade_log     = trade_log,
    )

    return bh, gq

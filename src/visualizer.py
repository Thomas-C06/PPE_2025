"""Data visualizer for GeoQuant AI -- Bloc 1 Data Engineering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend non-interactif pour la génération de fichiers PNG

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Événements géopolitiques/financiers clés à annoter sur les graphiques
_KEY_EVENTS: list[tuple[str, str, str]] = [
    ("2022-02-24", "Invasion\nUkraine",   "red"),
    ("2022-06-15", "Fed +75bp\n(1er)",    "#9467bd"),
    ("2022-11-11", "FTX\ncollapse",       "darkorange"),
    ("2023-03-10", "SVB\ncollapse",       "darkred"),
    ("2023-10-07", "Hamas\nattack",       "red"),
    ("2024-08-05", "Japan\ncrash",        "saddlebrown"),
    ("2024-09-18", "Fed cut\n-50bp",      "#2ca02c"),
    ("2024-11-05", "Trump\nelected",      "#1f77b4"),
]


@dataclass
class DataVisualizer:
    """
    Generate diagnostic charts for the GeoQuant AI Bloc 1 dataset.

    Charts produced:
        1. Price vs media activity (with geopolitical event annotations)
        2. Drawdown from running maximum (with max DD annotation)
        3. RSI 14-day (with overbought / oversold zones)
        4. Daily return distribution (with normal curve overlay)

    Attributes:
        base_dir: Base directory of the GeoQuant AI project.
        ticker_name: Human-readable name of the primary ticker (e.g. "SP500").
    """

    base_dir: Path
    ticker_name: str = "SP500"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _output_dir(self) -> Path:
        """Return and create the chart output directory."""
        charts_dir = self.base_dir / "data" / "processed" / "graphiques"
        charts_dir.mkdir(parents=True, exist_ok=True)
        return charts_dir

    @staticmethod
    def _price_col(df: pd.DataFrame) -> str:
        """Return the price column available in the DataFrame."""
        for col in ("Adj Close", "Close"):
            if col in df.columns:
                return col
        raise ValueError(
            f"No price column found. Available: {list(df.columns)}"
        )

    @staticmethod
    def _fmt_xaxis(ax: plt.Axes, fig: plt.Figure) -> None:
        """Apply consistent date formatting to the x-axis."""
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate(rotation=30)

    # ── Charts ────────────────────────────────────────────────────────────────

    def plot_price_vs_media(self, df: pd.DataFrame) -> Path:
        """
        Plot price history overlaid with daily news article count.

        Key geopolitical events are annotated with vertical dashed lines.

        Args:
            df: Final merged dataset.

        Returns:
            Path to the saved PNG file.
        """
        price_col = self._price_col(df)
        dates     = pd.to_datetime(df["date"])

        fig, ax1 = plt.subplots(figsize=(16, 7))

        # Courbe de prix (axe gauche)
        ax1.plot(
            dates, df[price_col],
            color="#1f77b4", linewidth=1.5, label=self.ticker_name,
        )
        ax1.set_ylabel("Price (USD)", color="#1f77b4", fontsize=11)
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        self._fmt_xaxis(ax1, fig)

        # Barres d'activité médiatique (axe droit)
        ax2 = ax1.twinx()
        ax2.bar(
            dates, df["nb_articles"],
            color="orange", alpha=0.35, width=1, label="News articles / day",
        )
        ax2.set_ylabel("Articles per day", color="darkorange", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="darkorange")

        # Annotations des événements clés
        price_max = df[price_col].max()
        price_min = df[price_col].min()
        y_text    = price_max - (price_max - price_min) * 0.05

        for evt_date_str, label, color in _KEY_EVENTS:
            evt_dt = pd.Timestamp(evt_date_str)
            if dates.min() <= evt_dt <= dates.max():
                ax1.axvline(
                    x=evt_dt, color=color,
                    linestyle="--", linewidth=0.9, alpha=0.75,
                )
                ax1.text(
                    evt_dt, y_text, label,
                    rotation=90, fontsize=7, color=color,
                    va="top", ha="right",
                )

        ax1.set_title(
            f"{self.ticker_name} -- Price vs. Media Activity (2022–2024)",
            fontsize=13, pad=12,
        )

        # Légende combinée
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

        plt.tight_layout()
        output = self._output_dir() / "01_price_vs_media.png"
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualizer] 01_price_vs_media.png  -> {output}")
        return output

    def plot_drawdown(self, df: pd.DataFrame) -> Path:
        """
        Plot the rolling drawdown from the all-time high.

        The maximum drawdown point is annotated on the chart.

        Args:
            df: Final merged dataset with a 'Drawdown' column.

        Returns:
            Path to the saved PNG file.
        """
        dates    = pd.to_datetime(df["date"])
        drawdown = df["Drawdown"]

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.fill_between(dates, drawdown * 100, 0, color="crimson", alpha=0.55)
        ax.plot(dates, drawdown * 100, color="crimson", linewidth=0.9, label="Drawdown")
        ax.axhline(0, color="black", linewidth=0.5)

        # Annotation du maximum drawdown
        max_dd_idx  = drawdown.idxmin()
        max_dd_val  = drawdown.iloc[max_dd_idx]
        max_dd_date = dates.iloc[max_dd_idx]
        offset_y    = max(3.0, abs(max_dd_val * 100) * 0.1)

        ax.annotate(
            f"Max DD: {max_dd_val:.1%}",
            xy=(max_dd_date, max_dd_val * 100),
            xytext=(max_dd_date, max_dd_val * 100 - offset_y),
            fontsize=9, color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
        )

        ax.set_ylabel("Drawdown (%)", fontsize=11)
        ax.set_title(
            f"{self.ticker_name} -- Drawdown from Running Maximum (2022–2024)",
            fontsize=13, pad=12,
        )
        self._fmt_xaxis(ax, fig)
        ax.legend(fontsize=9)

        plt.tight_layout()
        output = self._output_dir() / "02_drawdown.png"
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualizer] 02_drawdown.png         -> {output}")
        return output

    def plot_rsi(self, df: pd.DataFrame) -> Path:
        """
        Plot RSI with overbought (≥70) and oversold (≤30) zones highlighted.

        Args:
            df: Final merged dataset with a 'RSI_14' column.

        Returns:
            Path to the saved PNG file.
        """
        dates = pd.to_datetime(df["date"])
        rsi   = df["RSI_14"]

        fig, ax = plt.subplots(figsize=(14, 4))

        ax.plot(dates, rsi, color="#ff7f0e", linewidth=1.2, label="RSI 14")
        ax.axhline(70, color="red",   linestyle="--", linewidth=0.8, alpha=0.7, label="Overbought (70)")
        ax.axhline(30, color="green", linestyle="--", linewidth=0.8, alpha=0.7, label="Oversold (30)")
        ax.axhline(50, color="gray",  linestyle=":",  linewidth=0.6, alpha=0.5)

        ax.fill_between(dates, rsi, 70, where=(rsi >= 70), alpha=0.20, color="red")
        ax.fill_between(dates, rsi, 30, where=(rsi <= 30), alpha=0.20, color="green")

        ax.set_ylim(0, 100)
        ax.set_ylabel("RSI", fontsize=11)
        ax.set_title(
            f"{self.ticker_name} -- RSI 14-day (2022–2024)",
            fontsize=13, pad=12,
        )
        self._fmt_xaxis(ax, fig)
        ax.legend(fontsize=8, loc="upper right", ncol=2)

        plt.tight_layout()
        output = self._output_dir() / "03_rsi.png"
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualizer] 03_rsi.png              -> {output}")
        return output

    def plot_return_distribution(self, df: pd.DataFrame) -> Path:
        """
        Plot the distribution of daily returns with a normal PDF overlay.

        Also marks the mean and the 5% VaR on the chart.

        Args:
            df: Final merged dataset with a 'Returns' column.

        Returns:
            Path to the saved PNG file.
        """
        returns = df["Returns"].dropna() * 100  # en pourcentage

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(
            returns, bins=80, color="#1f77b4", alpha=0.65,
            density=True, label="Daily returns (%)",
        )

        # Courbe normale théorique
        mu, sigma = returns.mean(), returns.std()
        x_range   = np.linspace(returns.min(), returns.max(), 300)
        normal_pdf = (
            (1 / (sigma * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
        )
        ax.plot(
            x_range, normal_pdf,
            color="red", linewidth=1.5,
            label=f"Normal  μ={mu:.2f}%  σ={sigma:.2f}%",
        )

        # Lignes verticales : moyenne et VaR 5%
        var_5 = float(returns.quantile(0.05))
        ax.axvline(
            mu, color="orange", linewidth=1.2, linestyle="--",
            label=f"Mean = {mu:.2f}%",
        )
        ax.axvline(
            var_5, color="crimson", linewidth=1.2, linestyle=":",
            label=f"VaR 5% = {var_5:.2f}%",
        )

        ax.set_xlabel("Daily Return (%)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(
            f"{self.ticker_name} -- Daily Return Distribution (2022–2024)",
            fontsize=13, pad=12,
        )
        ax.legend(fontsize=9)

        plt.tight_layout()
        output = self._output_dir() / "04_return_distribution.png"
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualizer] 04_return_distribution.png -> {output}")
        return output

    # ── Orchestration ─────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> list[Path]:
        """
        Generate all four diagnostic charts.

        Args:
            df: Final merged dataset (output of DataMerger.run).

        Returns:
            List of paths to generated PNG files.
        """
        outputs: list[Path] = [
            self.plot_price_vs_media(df),
            self.plot_drawdown(df),
            self.plot_rsi(df),
            self.plot_return_distribution(df),
        ]
        print(f"  [Visualizer] {len(outputs)} charts saved to data/processed/graphiques/")
        return outputs

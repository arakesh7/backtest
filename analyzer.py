from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Analyzer(ABC):
    """
    Abstract base class for performance and risk analyzers.

    Analyzers are attached to a Backtester and are called at different
    points during the backtest to record data and calculate metrics.
    """
    def __init__(self):
        self.backtester = None  # Will be set by the backtester

    def attach_backtester(self, backtester):
        """Gives the analyzer a reference to the backtester instance."""
        self.backtester = backtester

    def on_start(self):
        """Called at the beginning of the backtest."""
        pass

    def on_bar(self, ts, data):
        """Called on each new bar."""
        pass

    def on_trade(self, trade):
        """Called when a trade is executed."""
        pass

    def on_end(self):
        """Called at the end of the backtest to perform final calculations."""
        pass

    @abstractmethod
    def get_results(self):
        """
        Return the calculated metrics as a dictionary.
        This is called after the backtest is finished.
        """
        raise NotImplementedError


class DefaultAnalyzer(Analyzer):
    """
    A default analyzer that calculates a standard set of performance metrics.
    """
    def __init__(self, risk_free_rate=0.0):
        super().__init__()
        self.trades = []
        self.equity_curve = []
        self.initial_cash = 0
        self.risk_free_rate = risk_free_rate

    def on_start(self):
        self.initial_cash = self.backtester._broker.cash

    def on_bar(self, ts, data):
        broker = self.backtester._broker
        # Calculate total portfolio value (cash + value of open positions)
        positions_value = sum(
            pos.size * data[pos.asset][4]  # pos.size * current_close_price
            for pos in broker.positions.values()
            if pos.asset in data and data[pos.asset] is not None
        )
        total_value = broker.cash + positions_value
        self.equity_curve.append((ts, total_value))

    def on_trade(self, trade):
        self.trades.append(trade)

    def get_results(self):
        if not self.equity_curve:
            return {"error": "No data to analyze."}

        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')
        equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)

        # --- PnL and Return ---
        final_equity = equity_df['equity'].iloc[-1]
        total_pnl = final_equity - self.initial_cash
        total_return_pct = (total_pnl / self.initial_cash) * 100

        # --- Trade Stats ---
        num_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        win_rate = (len(winning_trades) / num_trades) * 100 if num_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades and sum(t.pnl for t in losing_trades) != 0 else float('inf')
        max_profit_trade = max(t.pnl for t in winning_trades) if winning_trades else 0
        max_loss_trade = min(t.pnl for t in losing_trades) if losing_trades else 0

        # --- Risk Metrics ---
        # Assuming daily returns for annualization (252 trading days)
        # This is a simplification; a more accurate approach would resample to daily.
        trading_days_per_year = 252
        returns_std = equity_df['returns'].std()
        sharpe_ratio = (equity_df['returns'].mean() / returns_std) * np.sqrt(trading_days_per_year) if returns_std != 0 else 0

        # Max Drawdown: percentage drop from peak equity.
        # np.where guards against divide-by-zero if cumulative_max is ever 0.
        cumulative_max = equity_df['equity'].cummax()
        drawdown = np.where(cumulative_max > 0, (equity_df['equity'] - cumulative_max) / cumulative_max, 0)
        max_drawdown = np.min(drawdown) * 100

        return {
            "Initial Equity": self.initial_cash,
            "Final Equity": final_equity,
            "Total PnL": total_pnl,
            "Total Return (%)": total_return_pct,
            "Number of Trades": num_trades,
            "Win Rate (%)": win_rate,
            "Profit Factor": profit_factor,
            "Average Win": avg_win,
            "Average Loss": avg_loss,
            "Max Profit (Single Trade)": max_profit_trade,
            "Max Loss (Single Trade)": max_loss_trade,
            "Sharpe Ratio (Annualized)": sharpe_ratio,
            "Max Drawdown (%)": max_drawdown,
            "Equity Curve": equity_df
        }

'''
Oliver Marszalek
Momentum Trading Bot / Current Work in Progress
'''

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

price_cache = {}


def get_price(symbol, start_date, end_date):

    key = (symbol, start_date, end_date)

    if key in price_cache:
        return price_cache[key].copy()

    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if data is None or len(data) == 0:
        return pd.DataFrame()

    close = data['Close']

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    data = close.to_frame('Close')
    price_cache[key] = data.copy()

    return data


class MT_Model:

    def __init__(self, symbol, start_date, end_date, lookback_period=126, threshold=0.75,
                 vol_period=20, cost=0.001, allow_short=False, market_filter=True,
                 market_symbol='SPY', market_ma=200, stock_ma=100,
                 market_fast_ma=100, market_drop_limit=-0.08,
                 strategy_type='shock_momentum', shock_period=5,
                 shock_threshold=1.0, shock_weight=0.50):

        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.vol_period = vol_period
        self.cost = cost
        self.allow_short = allow_short
        self.market_filter = market_filter
        self.market_symbol = market_symbol
        self.market_ma = market_ma
        self.stock_ma = stock_ma
        self.market_fast_ma = market_fast_ma
        self.market_drop_limit = market_drop_limit
        self.strategy_type = strategy_type
        self.shock_period = shock_period
        self.shock_threshold = shock_threshold
        self.shock_weight = shock_weight
        self.data = None

    def fetch_data(self):

        warmup_days = max(450, self.lookback_period * 3, self.market_ma * 3, self.stock_ma * 3)
        fetch_start = pd.Timestamp(self.start_date) - pd.DateOffset(days=warmup_days)
        fetch_start = fetch_start.strftime('%Y-%m-%d')

        data = get_price(self.symbol, fetch_start, self.end_date)

        if len(data) == 0:
            print(f"Error fetching data for {self.symbol}")
            return

        self.data = data.copy()

        if self.market_filter:
            market = get_price(self.market_symbol, fetch_start, self.end_date)

            if len(market) > 0:
                market = market.rename(columns={'Close': 'Market_Close'})
                self.data = self.data.join(market, how='left')
                self.data['Market_Close'] = self.data['Market_Close'].ffill()

    def calculate_momentum(self):

        if self.data is None:
            print("Please fetch data first")
            return

        self.data['Returns'] = self.data['Close'].pct_change()

        self.data['Momentum'] = self.data['Close'].pct_change(periods=self.lookback_period)
        self.data['Vol'] = self.data['Returns'].rolling(self.vol_period).std() * np.sqrt(self.lookback_period)
        self.data['Score'] = self.data['Momentum'] / self.data['Vol']

        self.data['Shock_Return'] = self.data['Close'].pct_change(periods=self.shock_period)
        self.data['Shock_Vol'] = self.data['Returns'].rolling(self.vol_period).std() * np.sqrt(self.shock_period)
        self.data['Shock_Score'] = self.data['Shock_Return'] / self.data['Shock_Vol']

        self.data['Score'] = self.data['Score'].replace([np.inf, -np.inf], np.nan)
        self.data['Shock_Score'] = self.data['Shock_Score'].replace([np.inf, -np.inf], np.nan)

        self.data['Final_Score'] = self.data['Score']

        if self.strategy_type == 'shock_momentum':
            self.data['Final_Score'] = self.data['Score'] + self.shock_weight * self.data['Shock_Score']

        self.data['Stock_MA'] = self.data['Close'].rolling(self.stock_ma).mean()

        if self.strategy_type == 'shock_momentum':
            long_signal = (
                (
                    (self.data['Score'] > self.threshold) |
                    (
                        (self.data['Score'] > 0) &
                        (self.data['Shock_Score'] > self.shock_threshold) &
                        (self.data['Final_Score'] > self.threshold)
                    )
                ) &
                (self.data['Close'] > self.data['Stock_MA'])
            )

            short_signal = (
                (
                    (self.data['Score'] < -self.threshold) |
                    (
                        (self.data['Score'] < 0) &
                        (self.data['Shock_Score'] < -self.shock_threshold) &
                        (self.data['Final_Score'] < -self.threshold)
                    )
                ) &
                (self.data['Close'] < self.data['Stock_MA'])
            )
        else:
            long_signal = (
                (self.data['Score'] > self.threshold) &
                (self.data['Close'] > self.data['Stock_MA'])
            )

            short_signal = (
                (self.data['Score'] < -self.threshold) &
                (self.data['Close'] < self.data['Stock_MA'])
            )

        self.data['Signal'] = 0
        self.data['Signal'] = np.where(long_signal, 1, 0)

        if self.allow_short:
            self.data['Signal'] = np.where(short_signal, -1, self.data['Signal'])

        if self.market_filter and 'Market_Close' in self.data.columns:
            self.data['Market_MA'] = self.data['Market_Close'].rolling(self.market_ma).mean()
            self.data['Market_Fast_MA'] = self.data['Market_Close'].rolling(self.market_fast_ma).mean()
            self.data['Market_Return_20'] = self.data['Market_Close'].pct_change(20)

            self.data['Market_Trend'] = np.where(
                (self.data['Market_Close'] > self.data['Market_MA']) &
                (self.data['Market_Close'] > self.data['Market_Fast_MA']) &
                (self.data['Market_Return_20'] > self.market_drop_limit),
                1,
                -1
            )

            if self.allow_short:
                self.data['Signal'] = np.where(
                    (self.data['Signal'] == 1) & (self.data['Market_Trend'] != 1),
                    0,
                    self.data['Signal']
                )

                self.data['Signal'] = np.where(
                    (self.data['Signal'] == -1) & (self.data['Market_Trend'] != -1),
                    0,
                    self.data['Signal']
                )
            else:
                self.data['Signal'] = np.where(self.data['Market_Trend'] == 1, self.data['Signal'], 0)

        self.data['Position'] = self.data['Signal'].shift(1).fillna(0)
        self.data = self.data[self.data.index >= pd.Timestamp(self.start_date)].copy()

    def backtest(self, initial_capital=10000, plot=True):

        if self.data is None:
            print("Please fetch data and calculate momentum first")
            return

        if len(self.data) == 0:
            return

        self.data['Returns'] = self.data['Returns'].fillna(0)

        position_change = self.data['Position'].diff().abs().fillna(0)

        if len(position_change) > 0:
            position_change.iat[0] = abs(self.data['Position'].iloc[0])

        self.data['Cost'] = position_change * self.cost
        self.data['Strategy_Returns'] = self.data['Returns'] * self.data['Position'] - self.data['Cost']

        self.data['Cumulative_Market'] = (1 + self.data['Returns']).cumprod()
        self.data['Cumulative_Strategy'] = (1 + self.data['Strategy_Returns']).cumprod()

        self.data['Market_Value'] = initial_capital * self.data['Cumulative_Market']
        self.data['Strategy_Value'] = initial_capital * self.data['Cumulative_Strategy']

        total_return = (self.data['Strategy_Value'].iloc[-1] / initial_capital - 1) * 100
        market_return = (self.data['Market_Value'].iloc[-1] / initial_capital - 1) * 100

        daily_rf = .02 / 252
        excess_returns = self.data['Strategy_Returns'] - daily_rf

        if excess_returns.std() == 0 or np.isnan(excess_returns.std()):
            sharpe_ratio = np.nan
        else:
            sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

        rolling_max = self.data['Strategy_Value'].expanding().max()
        drawdown = (self.data['Strategy_Value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        if plot:
            print("\n=== Backtest Results ===")
            print(f"Symbol: {self.symbol}")
            print(f"Period: {self.start_date} to {self.end_date}")
            print(f"Final Strategy Value: ${self.data['Strategy_Value'].iloc[-1]:,.2f}")
            print(f"Final Market Value: ${self.data['Market_Value'].iloc[-1]:,.2f}")
            print(f"Strategy Return: {total_return:.2f}%")
            print(f"Market Return: {market_return:.2f}%")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2f}%")

        return {
            'total_return': total_return,
            'market_return': market_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'Strategy_Returns': self.data['Strategy_Returns'],
            'Position': self.data['Position']
        }

    def plot_results(self):

        if self.data is None:
            print("Please run backtest first")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        ax1.plot(self.data['Close'], label='Price', alpha=0.7)
        buy_signals = self.data[self.data['Signal'] == 1]
        sell_signals = self.data[self.data['Signal'] == -1]
        ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', s=100, label='Buy Signal', alpha=0.7)
        ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', s=100, label='Sell Signal', alpha=0.7)
        ax1.set_title(f'{self.symbol} Price and Trading Signals')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.data['Final_Score'], label='Final Momentum Score', alpha=0.7)
        ax2.axhline(y=self.threshold, linestyle='--', alpha=0.7, label='Buy Threshold')

        if self.allow_short:
            ax2.axhline(y=-self.threshold, linestyle='--', alpha=0.7, label='Short Threshold')

        ax2.axhline(y=0, linestyle='-', alpha=0.5)
        ax2.set_title('Momentum Score')
        ax2.legend()
        ax2.grid(True)

        ax3.plot(self.data['Market_Value'], label='Buy & Hold', alpha=0.7)
        ax3.plot(self.data['Strategy_Value'], label='Momentum Strategy', alpha=0.7)
        ax3.set_title('Portfolio Value Comparison')
        ax3.set_ylabel('Portfolio Value ($)')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

    def run_complete_backtest(self, initial_capital=10000, plot=True):

        self.fetch_data()

        if self.data is not None:
            self.calculate_momentum()
            results = self.backtest(initial_capital, plot)

            if plot:
                self.plot_results()

            return results
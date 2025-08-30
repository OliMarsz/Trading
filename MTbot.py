'''
Oliver Marszalek
08/15/2025
Begginer Momentum Trading Bot / Current Work in Progress
'''

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class MT_Model:

    def __init__(self, symbol, start_date, end_date, lookback_period=20, threshold=0.02):

        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.data = None
        self.signals = None
        self.returns = None
        
    def fetch_data(self):
        
        try:
            data = yf.download(self.symbol, start=self.start_date, end=self.end_date, auto_adjust=True)
            self.data = data[['Close']].copy()
            print(f"Successfully fetched data for {self.symbol}")
        except Exception as e:
            print(f"Error fetching data: {e}")
            
    def calculate_momentum(self):
        
        if self.data is None:
            print("Please fetch data first")
            return
            
        self.data['Returns'] = self.data['Close'].pct_change()
        
        self.data['Momentum'] = self.data['Close'].pct_change(periods=self.lookback_period)
        
        self.data['Signal'] = 0
        self.data['Signal'] = np.where(self.data['Momentum'] > self.threshold, 1, 0)
        self.data['Signal'] = np.where(self.data['Momentum'] < -self.threshold, -1, self.data['Signal'])
        
        self.data['Position'] = self.data['Signal'].shift(1)
        
    def backtest(self, initial_capital=10000, plot=True):
       
        if self.data is None:
            print("Please fetch data and calculate momentum first")
            return
            
        self.data['Cost'] = self.data['Position'].diff().abs() * 0.001
            
        self.data['Strategy_Returns'] = self.data['Returns'] * self.data['Position'] - self.data['Cost']
        
        self.data['Cumulative_Market'] = (1 + self.data['Returns']).cumprod()
        self.data['Cumulative_Strategy'] = (1 + self.data['Strategy_Returns']).cumprod()
        
        self.data['Market_Value'] = initial_capital * self.data['Cumulative_Market']
        self.data['Strategy_Value'] = initial_capital * self.data['Cumulative_Strategy']
        
        total_return = (self.data['Strategy_Value'].iloc[-1] / initial_capital - 1) * 100
        market_return = (self.data['Market_Value'].iloc[-1] / initial_capital - 1) * 100

        daily_rf = .02/252
        excess_returns = self.data['Strategy_Returns'] - daily_rf
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        
        rolling_max = self.data['Strategy_Value'].expanding().max()
        drawdown = (self.data['Strategy_Value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        if plot:
        
            print("\n=== Backtest Results ===")
            print(f"Symbol: {self.symbol}")
            print(f"Period: {self.start_date} to {self.end_date}")
            print(f"Initial Capital: ${initial_capital:,.2f}")
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
            'Strategy_Returns': self.data['Strategy_Returns']
        }
    
    def plot_results(self):
        
        if self.data is None:
            print("Please run backtest first")
            return
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        ax1.plot(self.data['Close'], label='Price', alpha=0.7)
        buy_signals = self.data[self.data['Signal'] == 1]
        sell_signals = self.data[self.data['Signal'] == -1]
        ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', 
                   marker='^', s=100, label='Buy Signal', alpha=0.7)
        ax1.scatter(sell_signals.index, sell_signals['Close'], color='red', 
                   marker='v', s=100, label='Sell Signal', alpha=0.7)
        ax1.set_title(f'{self.symbol} Price and Trading Signals')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.data['Momentum'], label='Momentum', color='purple', alpha=0.7)
        ax2.axhline(y=self.threshold, color='green', linestyle='--', alpha=0.7, label='Buy Threshold')
        ax2.axhline(y=-self.threshold, color='red', linestyle='--', alpha=0.7, label='Sell Threshold')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Momentum Indicator')
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
        """Run complete backtest pipeline"""
        self.fetch_data()
        if self.data is not None:
            self.calculate_momentum()
            results = self.backtest(initial_capital, plot)
            if plot:
                self.plot_results()
            return results





        


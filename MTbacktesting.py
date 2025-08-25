import MTbot
import math
import numpy as np

symbol_list = ['AAPL', 'MSFT', 'UNH', 'JNJ', 'V', 'JPM', 'WMT', 'PG', 'XOM',
               'CVX', 'HD', 'LLY', 'BAC', 'ABBV', 'KO', 'PFE', 'AVGO', 'DIS', 'PEP',
               'CSCO', 'VZ', 'CMCSA', 'MCD', 'NKE', 'MRK', 'AMGN', 'DOW', 'HON', 'CAT',
               'IBM', 'GS', 'MMM', 'WBA', 'TRV']
short_symbol_list = ['AAPL', 'MSFT', 'UNH']

def find_sharpe(daily_returns):
    
    daily_rf = .02/252
    excess_returns = daily_returns - daily_rf

    sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    return sharpe

def optimize_returns(symbols, train_start="2020-01-01", train_end="2022-12-31", 
                     test_start="2023-01-01", test_end="2025-01-01"):

    best_ratio = -np.inf
    best_lookback = None
    best_threshold = None

    for x in range(1, 366, 5):
        for y in np.arange(0.01, 0.11, 0.01):
            if x < 10 and y > 0.05:
                continue
            if x > 200 and y < 0.03:
                continue
            
            returns = []
            for stock in symbols:
                model = MTbot.MT_Model(
                    symbol=stock,
                    start_date=train_start,
                    end_date=train_end,
                    lookback_period=x,
                    threshold=y
                )
                results = model.run_complete_backtest(plot=False)
                daily_returns = results['Strategy_Returns'].dropna()
                returns.extend(daily_returns.tolist())

            daily_returns = np.array(returns)
            sharpe_ratio = find_sharpe(daily_returns)

            if sharpe_ratio > best_ratio:
                best_ratio = sharpe_ratio
                best_lookback = x
                best_threshold = y

    print(f"Sharpe Ratio: {best_ratio:.2f}")
    print(f"Best Lookback: {best_lookback}")
    print(f"Best Threshold: {best_threshold}")

    test_returns = []
    for stock in symbols:
        model = MTbot.MT_Model(
            symbol=stock,
            start_date=test_start,
            end_date=test_end,
            lookback_period=best_lookback,
            threshold=best_threshold
        )
        results = model.run_complete_backtest(plot=False)
        daily_returns = results['Strategy_Returns'].dropna()
        test_returns.extend(daily_returns.tolist())

    test_returns = np.array(test_returns)
    test_sharpe = find_sharpe(test_returns)
    cumulative = np.cumprod(1 + test_returns)[-1] - 1

    print(f"Final Return: {cumulative*100:.2f}%")
    print(f"Sharpe Ratio: {test_sharpe:.2f}")

    return best_lookback, best_threshold

optimize_returns(short_symbol_list)





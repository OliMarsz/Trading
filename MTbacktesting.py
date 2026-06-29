'''
Oliver Marszalek
Backtesting Training and Testing Period / Current Work in Progress
'''

import MTbot
import numpy as np
import pandas as pd

symbol_list = ['AAPL', 'MSFT', 'UNH', 'JNJ', 'V', 'JPM', 'WMT', 'PG', 'XOM',
               'CVX', 'HD', 'LLY', 'BAC', 'ABBV', 'KO', 'PFE', 'AVGO', 'DIS', 'PEP',
               'CSCO', 'VZ', 'CMCSA', 'MCD', 'NKE', 'MRK', 'AMGN', 'DOW', 'HON', 'CAT',
               'IBM', 'GS', 'MMM', 'WBA', 'TRV']

short_symbol_list = ['AAPL', 'MSFT', 'UNH']


def get_portfolio_returns(symbols, start_date, end_date, lookback_period, threshold,
                          vol_period=20, allow_short=False, market_filter=True,
                          stock_ma=100):

    returns = []
    positions = []

    for stock in symbols:
        model = MTbot.MT_Model(
            symbol=stock,
            start_date=start_date,
            end_date=end_date,
            lookback_period=lookback_period,
            threshold=threshold,
            vol_period=vol_period,
            allow_short=allow_short,
            market_filter=market_filter,
            stock_ma=stock_ma
        )

        results = model.run_complete_backtest(plot=False)

        if results is None:
            continue

        daily_returns = results['Strategy_Returns'].dropna().rename(stock)
        daily_position = results['Position'].abs().reindex(daily_returns.index).rename(stock)

        returns.append(daily_returns)
        positions.append(daily_position)

    if len(returns) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    returns = pd.concat(returns, axis=1)
    positions = pd.concat(positions, axis=1)

    portfolio_returns = returns.mean(axis=1).dropna()
    portfolio_exposure = positions.mean(axis=1).reindex(portfolio_returns.index).fillna(0)

    return portfolio_returns, portfolio_exposure


def find_stats(daily_returns, exposure=None):

    daily_returns = pd.Series(daily_returns).dropna()

    if len(daily_returns) < 2:
        return None

    daily_rf = .02 / 252
    excess_returns = daily_returns - daily_rf

    total_return = (1 + daily_returns).prod() - 1
    years = len(daily_returns) / 252

    if total_return <= -1:
        cagr = np.nan
    else:
        cagr = (1 + total_return) ** (1 / years) - 1

    annual_vol = daily_returns.std() * np.sqrt(252)

    if excess_returns.std() == 0 or np.isnan(excess_returns.std()):
        sharpe = np.nan
    else:
        sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()

    if downside_std == 0 or np.isnan(downside_std):
        sortino = np.nan
    else:
        sortino = np.sqrt(252) * (excess_returns.mean() / downside_std)

    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    if max_drawdown == 0 or np.isnan(max_drawdown):
        calmar = np.nan
    else:
        calmar = cagr / abs(max_drawdown)

    win_rate = (daily_returns > 0).mean()
    best_day = daily_returns.max()
    worst_day = daily_returns.min()

    if exposure is None or len(exposure) == 0:
        avg_exposure = np.nan
    else:
        avg_exposure = pd.Series(exposure).reindex(daily_returns.index).fillna(0).mean()

    return {
        'total_return': total_return,
        'cagr': cagr,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'win_rate': win_rate,
        'best_day': best_day,
        'worst_day': worst_day,
        'avg_exposure': avg_exposure
    }


def print_stats(name, stats):

    if stats is None:
        print(f"\nNo stats for {name}")
        return

    print(f"\n=== {name} Stats ===")
    print(f"Total Return: {stats['total_return'] * 100:.2f}%")
    print(f"CAGR: {stats['cagr'] * 100:.2f}%")
    print(f"Annual Volatility: {stats['annual_vol'] * 100:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe']:.2f}")
    print(f"Sortino Ratio: {stats['sortino']:.2f}")
    print(f"Max Drawdown: {stats['max_drawdown'] * 100:.2f}%")
    print(f"Calmar Ratio: {stats['calmar']:.2f}")
    print(f"Win Rate: {stats['win_rate'] * 100:.2f}%")
    print(f"Best Day: {stats['best_day'] * 100:.2f}%")
    print(f"Worst Day: {stats['worst_day'] * 100:.2f}%")
    print(f"Average Exposure: {stats['avg_exposure'] * 100:.2f}%")


def score_stats(stats):

    if stats is None:
        return -np.inf

    if np.isnan(stats['sharpe']):
        return -np.inf

    if np.isnan(stats['avg_exposure']):
        return -np.inf

    if stats['avg_exposure'] < 0.10:
        return -np.inf

    if stats['max_drawdown'] < -0.25:
        return -np.inf

    score = stats['sharpe']

    if not np.isnan(stats['calmar']):
        score += stats['calmar'] * 0.25

    if stats['max_drawdown'] < -0.15:
        score -= 0.50

    if stats['annual_vol'] > 0.30:
        score -= 0.50

    if stats['avg_exposure'] < 0.20:
        score -= 0.25

    return score


def optimize_params(symbols, train_start, train_end, allow_short=False, market_filter=True):

    best_score = -np.inf
    best_lookback = None
    best_threshold = None
    best_vol_period = None
    best_stock_ma = None
    best_stats = None

    lookback_grid = [63, 126, 252]
    threshold_grid = [0.75, 1.00, 1.25, 1.50]
    vol_grid = [20, 60]
    stock_ma_grid = [50, 100, 200]

    for lookback in lookback_grid:
        for threshold in threshold_grid:
            for vol_period in vol_grid:
                for stock_ma in stock_ma_grid:

                    train_returns, train_exposure = get_portfolio_returns(
                        symbols=symbols,
                        start_date=train_start,
                        end_date=train_end,
                        lookback_period=lookback,
                        threshold=threshold,
                        vol_period=vol_period,
                        allow_short=allow_short,
                        market_filter=market_filter,
                        stock_ma=stock_ma
                    )

                    stats = find_stats(train_returns, train_exposure)
                    score = score_stats(stats)

                    if score > best_score:
                        best_score = score
                        best_lookback = lookback
                        best_threshold = threshold
                        best_vol_period = vol_period
                        best_stock_ma = stock_ma
                        best_stats = stats

    if best_lookback is None:
        return None

    return best_lookback, best_threshold, best_vol_period, best_stock_ma, best_stats


def make_windows(start_date, end_date, train_years=2, test_months=6):

    train_start = pd.Timestamp(start_date)
    final_end = pd.Timestamp(end_date)

    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_start >= final_end:
            break

        if test_end > final_end:
            test_end = final_end

        yield train_start, train_end, test_start, test_end

        if test_end >= final_end:
            break

        train_start = train_start + pd.DateOffset(months=test_months)


def date_string(date):
    return pd.Timestamp(date).strftime('%Y-%m-%d')


def walk_forward_test(symbols, start_date="2018-01-01", end_date="2025-01-01",
                      train_years=2, test_months=6, allow_short=False, market_filter=True):

    all_test_returns = []
    all_test_exposure = []
    param_history = []

    for train_start, train_end, test_start, test_end in make_windows(
        start_date=start_date,
        end_date=end_date,
        train_years=train_years,
        test_months=test_months
    ):

        train_start = date_string(train_start)
        train_end = date_string(train_end)
        test_start = date_string(test_start)
        test_end = date_string(test_end)

        best = optimize_params(
            symbols=symbols,
            train_start=train_start,
            train_end=train_end,
            allow_short=allow_short,
            market_filter=market_filter
        )

        if best is None:
            continue

        lookback, threshold, vol_period, stock_ma, train_stats = best

        test_returns, test_exposure = get_portfolio_returns(
            symbols=symbols,
            start_date=test_start,
            end_date=test_end,
            lookback_period=lookback,
            threshold=threshold,
            vol_period=vol_period,
            allow_short=allow_short,
            market_filter=market_filter,
            stock_ma=stock_ma
        )

        test_stats = find_stats(test_returns, test_exposure)

        if test_stats is None:
            continue

        all_test_returns.append(test_returns)
        all_test_exposure.append(test_exposure)

        param_history.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'lookback': lookback,
            'threshold': threshold,
            'vol_period': vol_period,
            'stock_ma': stock_ma,
            'train_sharpe': train_stats['sharpe'],
            'test_sharpe': test_stats['sharpe'],
            'test_return': test_stats['total_return'],
            'test_drawdown': test_stats['max_drawdown'],
            'test_exposure': test_stats['avg_exposure']
        })

        print(f"\nTrain: {train_start} to {train_end}")
        print(f"Test: {test_start} to {test_end}")
        print(f"Lookback: {lookback}")
        print(f"Threshold: {threshold}")
        print(f"Vol Period: {vol_period}")
        print(f"Stock MA: {stock_ma}")
        print(f"Train Sharpe: {train_stats['sharpe']:.2f}")
        print(f"Test Sharpe: {test_stats['sharpe']:.2f}")
        print(f"Test Return: {test_stats['total_return'] * 100:.2f}%")
        print(f"Test Drawdown: {test_stats['max_drawdown'] * 100:.2f}%")
        print(f"Test Exposure: {test_stats['avg_exposure'] * 100:.2f}%")

    if len(all_test_returns) == 0:
        print("No test returns")
        return None, None

    final_returns = pd.concat(all_test_returns).sort_index()
    final_exposure = pd.concat(all_test_exposure).sort_index()

    history = pd.DataFrame(param_history)

    final_stats = find_stats(final_returns, final_exposure)
    print_stats("Walk Forward", final_stats)

    print("\n=== Parameter History ===")
    print(history)

    return final_returns, history


walk_forward_test(short_symbol_list)
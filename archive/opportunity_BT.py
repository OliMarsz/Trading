'''
Oliver Marszalek
Opportunistic Shock Momentum Prototype
'''

import numpy as np
import pandas as pd
import yfinance as yf

pd.set_option('display.max_columns', None)

popular_universe = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'AVGO', 'TSLA', 'BRK-B',
    'JPM', 'LLY', 'V', 'XOM', 'UNH', 'MA', 'COST', 'WMT', 'HD', 'PG',
    'NFLX', 'JNJ', 'ABBV', 'BAC', 'KO', 'CRM', 'ORCL', 'AMD', 'CVX', 'MRK',
    'PEP', 'LIN', 'ADBE', 'TMO', 'MCD', 'CSCO', 'ACN', 'ABT', 'IBM', 'GE',
    'QCOM', 'WFC', 'TXN', 'DHR', 'VZ', 'INTU', 'AMGN', 'NOW', 'PM', 'ISRG',
    'AMAT', 'NEE', 'RTX', 'PFE', 'LOW', 'SPGI', 'UBER', 'CAT', 'GS', 'UNP',
    'PGR', 'HON', 'BKNG', 'BLK', 'SYK', 'TJX', 'T', 'ETN', 'LMT', 'VRTX',
    'AXP', 'COP', 'BSX', 'PANW', 'C', 'ADP', 'MDT', 'CB', 'ADI', 'MU',
    'DE', 'PLD', 'SBUX', 'GILD', 'CI', 'SCHW', 'SO', 'MO',
    'ELV', 'DUK', 'REGN', 'ZTS', 'BA', 'KLAC', 'ICE', 'SHW', 'EQIX', 'MCO',
    'CME', 'WM', 'AON', 'CDNS', 'SNPS', 'APH', 'HCA', 'CL', 'CMG', 'ITW',
    'GD', 'NOC', 'MSI', 'EOG', 'APD', 'PH', 'USB', 'MAR', 'MMM', 'PNC',
    'TDG', 'ORLY', 'FCX', 'EMR', 'ROP', 'AJG', 'NSC', 'NXPI', 'ECL', 'FDX',
    'TGT', 'PSX', 'MPC', 'AFL', 'GM', 'F', 'AZO', 'PCAR', 'COF', 'OXY',
    'HLT', 'ROST', 'TRV', 'AIG', 'MET', 'ALL', 'KMB', 'KR', 'DLTR', 'DHI',
    'LEN', 'NEM', 'AEP', 'EXC', 'SRE', 'XEL', 'WELL', 'O', 'SPG', 'PSA',
    'DLR', 'CCI', 'AMT', 'PLTR', 'SHOP', 'SNOW', 'DDOG', 'NET', 'CRWD', 'ZS',
    'MDB', 'TEAM', 'ROKU', 'PYPL', 'COIN', 'RBLX', 'U', 'DASH', 'ABNB',
    'MRVL', 'LRCX', 'MCHP', 'ON', 'MPWR', 'ASML', 'TSM', 'ARM', 'SMCI', 'DELL',
    'HPQ', 'ANET', 'FTNT', 'OKTA', 'WDAY', 'ADSK', 'TTD', 'APP', 'MELI', 'SE',
    'BABA', 'PDD', 'JD', 'NKE', 'LULU', 'EL', 'ULTA', 'CAVA', 'CMI', 'URI'
]


def download_data(symbols, start_date, end_date, market_symbol='SPY', chunk_size=50):

    symbols = list(dict.fromkeys(symbols + [market_symbol]))

    warmup_start = pd.Timestamp(start_date) - pd.DateOffset(days=600)
    warmup_start = warmup_start.strftime('%Y-%m-%d')

    close_list = []
    volume_list = []

    for i in range(0, len(symbols), chunk_size):
        batch = symbols[i:i + chunk_size]

        data = yf.download(
            batch,
            start=warmup_start,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by='column',
            threads=True
        )

        if data is None or len(data) == 0:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' not in data.columns.get_level_values(0):
                continue

            close = data['Close'].copy()
            volume = data['Volume'].copy()
        else:
            if 'Close' not in data.columns:
                continue

            ticker = batch[0]
            close = data['Close'].to_frame(ticker)
            volume = data['Volume'].to_frame(ticker)

        close_list.append(close)
        volume_list.append(volume)

    if len(close_list) == 0:
        return None, None

    close = pd.concat(close_list, axis=1)
    volume = pd.concat(volume_list, axis=1)

    close = close.loc[:, ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]

    close = close.dropna(axis=1, how='all')
    volume = volume.reindex(columns=close.columns)

    return close, volume


def find_stats(daily_returns, exposure=None, rf_rate=0.02):

    daily_returns = pd.Series(daily_returns).dropna()

    if len(daily_returns) < 2:
        return None

    total_return = (1 + daily_returns).prod() - 1
    years = len(daily_returns) / 252

    if total_return <= -1:
        cagr = np.nan
    else:
        cagr = (1 + total_return) ** (1 / years) - 1

    annual_vol = daily_returns.std() * np.sqrt(252)

    daily_rf = rf_rate / 252
    excess_returns = daily_returns - daily_rf

    if excess_returns.std() == 0 or np.isnan(excess_returns.std()):
        sharpe = np.nan
    else:
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    downside_returns = daily_returns[daily_returns < 0]

    if downside_returns.std() == 0 or np.isnan(downside_returns.std()):
        sortino = np.nan
    else:
        sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()

    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    if max_drawdown == 0 or np.isnan(max_drawdown):
        calmar = np.nan
    else:
        calmar = cagr / abs(max_drawdown)

    if exposure is None:
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
        'win_rate': (daily_returns > 0).mean(),
        'best_day': daily_returns.max(),
        'worst_day': daily_returns.min(),
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


def cap_exposure(weights, max_gross_exposure=1.00):

    gross = weights.abs().sum(axis=1)

    scale = max_gross_exposure / gross
    scale = scale.where(gross > max_gross_exposure, 1)
    scale = scale.replace([np.inf, -np.inf], 1).fillna(1)

    return weights.mul(scale, axis=0)


def exact_hold_positions(entry_weights, hold_days, max_position_size, max_gross_exposure):

    positions = entry_weights * 0

    for lag in range(1, hold_days + 1):
        positions = positions.add(entry_weights.shift(lag), fill_value=0)

    positions = positions.clip(lower=0, upper=max_position_size)
    positions = cap_exposure(positions, max_gross_exposure)

    return positions.fillna(0)


def signal_diagnostics(selected, prices, horizons=[1, 2, 3, 5, 10]):

    print("\n=== Signal Diagnostics ===")

    for h in horizons:
        future_return = prices.shift(-h) / prices - 1
        signal_returns = future_return.where(selected).stack().dropna()

        if len(signal_returns) == 0:
            print(f"{h}D Forward Return: No signals")
        else:
            print(f"{h}D Forward Return: {signal_returns.mean() * 100:.2f}% avg, "
                  f"{(signal_returns > 0).mean() * 100:.2f}% win rate, "
                  f"{len(signal_returns)} samples")


def score_bucket_diagnostics(selected, final_score, prices, horizons=[3, 5, 10], buckets=5):

    print("\n=== Score Bucket Diagnostics ===")

    selected_scores = final_score.where(selected).stack().dropna()
    selected_scores.name = 'score'

    if len(selected_scores) < buckets:
        print("Not enough selected signals for bucket diagnostics")
        return

    for h in horizons:
        future_return = prices.shift(-h) / prices - 1
        signal_returns = future_return.where(selected).stack().dropna()
        signal_returns.name = 'forward_return'

        data = pd.concat([selected_scores, signal_returns], axis=1).dropna()

        if len(data) < buckets:
            print(f"{h}D Buckets: Not enough data")
            continue

        data['bucket'] = pd.qcut(
            data['score'],
            q=buckets,
            labels=False,
            duplicates='drop'
        )

        print(f"\n{h}D Forward Return By Score Bucket")

        grouped = data.groupby('bucket')

        for bucket, group in grouped:
            score_min = group['score'].min()
            score_max = group['score'].max()
            avg_return = group['forward_return'].mean()
            win_rate = (group['forward_return'] > 0).mean()
            count = len(group)

            print(f"Bucket {int(bucket) + 1}: "
                  f"score {score_min:.2f} to {score_max:.2f}, "
                  f"avg {avg_return * 100:.2f}%, "
                  f"win {win_rate * 100:.2f}%, "
                  f"n={count}")


def run_opportunistic_backtest(symbols=None, start_date='2018-01-01', end_date='2025-01-01',
                               market_symbol='SPY', lookback=63, long_lookback=126,
                               shock_period=3, vol_period=20,
                               stock_fast_ma=20, stock_slow_ma=50,
                               market_fast_ma=50, market_slow_ma=100,
                               breakout_lookback=20,
                               min_price=10, min_dollar_volume=50000000,
                               min_base_score=0.10, min_long_score=0.00,
                               min_shock_score=3.00, min_final_score=3.25,
                               min_score_percentile=0.98,
                               min_shock_return=0.03, max_shock_return=0.12,
                               max_extension=0.10, shock_weight=0.75,
                               max_positions=5, max_position_size=0.05,
                               max_gross_exposure=1.00, strength_scale=1.50,
                               hold_days=5, cost=0.001, rf_rate=0.02):

    if symbols is None:
        symbols = popular_universe

    close, volume = download_data(symbols, start_date, end_date, market_symbol)

    if close is None:
        print("No data downloaded")
        return None

    symbols = [s for s in symbols if s in close.columns and s != market_symbol]

    if market_symbol not in close.columns:
        print("Market symbol not found")
        return None

    prices = close[symbols].copy()
    volumes = volume[symbols].copy()

    returns = prices.pct_change().fillna(0)

    dollar_volume = prices * volumes
    avg_dollar_volume = dollar_volume.rolling(20).mean()

    vol = returns.rolling(vol_period).std()

    momentum = prices.pct_change(lookback)
    long_momentum = prices.pct_change(long_lookback)

    base_score = momentum / (vol * np.sqrt(lookback))
    long_score = long_momentum / (vol * np.sqrt(long_lookback))

    shock_return = prices.pct_change(shock_period)
    shock_score = shock_return / (vol * np.sqrt(shock_period))

    final_score = base_score + shock_weight * shock_score

    base_score = base_score.replace([np.inf, -np.inf], np.nan)
    long_score = long_score.replace([np.inf, -np.inf], np.nan)
    shock_score = shock_score.replace([np.inf, -np.inf], np.nan)
    final_score = final_score.replace([np.inf, -np.inf], np.nan)

    score_percentile = final_score.rank(axis=1, pct=True)

    stock_fast = prices.rolling(stock_fast_ma).mean()
    stock_slow = prices.rolling(stock_slow_ma).mean()

    extension = prices / stock_fast - 1

    prior_high = prices.shift(1).rolling(breakout_lookback).max()
    breakout = prices > prior_high

    stock_regime = (
        (prices > stock_fast) &
        (prices > stock_slow) &
        (stock_fast > stock_slow) &
        (momentum > 0) &
        (long_momentum > 0) &
        (base_score > min_base_score) &
        (long_score > min_long_score) &
        (extension < max_extension) &
        (prices > min_price) &
        (avg_dollar_volume > min_dollar_volume)
    )

    spy = close[market_symbol].copy()
    spy_fast = spy.rolling(market_fast_ma).mean()
    spy_slow = spy.rolling(market_slow_ma).mean()
    spy_return_5 = spy.pct_change(5)
    spy_return_20 = spy.pct_change(20)

    market_regime = (
        (spy > spy_fast) &
        (spy > spy_slow) &
        (spy_fast > spy_slow) &
        (spy_return_5 > -0.03) &
        (spy_return_20 > -0.05)
    )

    shock_setup = (
        breakout &
        (shock_score > min_shock_score) &
        (shock_return > min_shock_return) &
        (shock_return < max_shock_return) &
        (final_score > min_final_score) &
        (score_percentile >= min_score_percentile)
    )

    eligible = stock_regime & shock_setup
    eligible = eligible.mul(market_regime, axis=0).astype(bool)

    eligible_scores = final_score.where(eligible)
    ranks = eligible_scores.rank(axis=1, ascending=False, method='first')

    selected = ranks <= max_positions
    selected_scores = final_score.where(selected)

    strength = (selected_scores - min_final_score) / strength_scale
    strength = strength.clip(lower=0, upper=1)
    strength = strength.fillna(0)

    entry_weights = strength * max_position_size
    entry_weights = cap_exposure(entry_weights, max_gross_exposure)

    positions = exact_hold_positions(
        entry_weights=entry_weights,
        hold_days=hold_days,
        max_position_size=max_position_size,
        max_gross_exposure=max_gross_exposure
    )

    market_exit = (
        (spy > spy_fast) &
        (spy_return_5 > -0.04)
    )

    stock_exit = prices > stock_fast

    exit_ok = stock_exit.mul(market_exit, axis=0)
    exit_mask = exit_ok.shift(1).fillna(False).infer_objects(copy=False).astype(bool)

    positions = positions.where(exit_mask, 0)
    positions = positions.fillna(0)
    positions = cap_exposure(positions, max_gross_exposure)

    exposure = positions.abs().sum(axis=1)

    turnover = positions.diff().abs().sum(axis=1).fillna(0)

    if len(turnover) > 0:
        turnover.iat[0] = positions.iloc[0].abs().sum()

    daily_rf = rf_rate / 252

    stock_return = (positions * returns).sum(axis=1)
    cash_return = (1 - exposure).clip(lower=0) * daily_rf

    strategy_returns = stock_return + cash_return - turnover * cost

    strategy_returns = strategy_returns[strategy_returns.index >= pd.Timestamp(start_date)]
    exposure = exposure.reindex(strategy_returns.index).fillna(0)

    spy_returns = spy.pct_change().reindex(strategy_returns.index).fillna(0)
    universe_returns = returns.mean(axis=1).reindex(strategy_returns.index).fillna(0)

    matched_spy_returns = exposure * spy_returns + (1 - exposure).clip(lower=0) * daily_rf
    matched_universe_returns = exposure * universe_returns + (1 - exposure).clip(lower=0) * daily_rf

    strategy_stats = find_stats(strategy_returns, exposure, rf_rate)
    spy_stats = find_stats(spy_returns, pd.Series(1, index=spy_returns.index), rf_rate)
    universe_stats = find_stats(universe_returns, pd.Series(1, index=universe_returns.index), rf_rate)
    matched_spy_stats = find_stats(matched_spy_returns, exposure, rf_rate)
    matched_universe_stats = find_stats(matched_universe_returns, exposure, rf_rate)

    print_stats("Opportunistic Shock Momentum", strategy_stats)
    print_stats("SPY Buy and Hold", spy_stats)
    print_stats("Equal Weight Universe", universe_stats)
    print_stats("Exposure Matched SPY", matched_spy_stats)
    print_stats("Exposure Matched Universe", matched_universe_stats)

    signal_count = selected.sum(axis=1)
    signal_count = signal_count[signal_count.index >= pd.Timestamp(start_date)]

    print("\n=== Trading Activity ===")
    print(f"Universe Size: {len(symbols)}")
    print(f"Total Signal Days: {(signal_count > 0).sum()}")
    print(f"Average Signals Per Day: {signal_count.mean():.2f}")
    print(f"Max Signals In One Day: {signal_count.max()}")
    print(f"Percent Days With Exposure: {(exposure > 0).mean() * 100:.2f}%")
    print(f"Hold Days: {hold_days}")

    selected_test = selected[selected.index >= pd.Timestamp(start_date)]

    signal_diagnostics(selected_test, prices)
    score_bucket_diagnostics(selected_test, final_score, prices)

    latest_date = final_score.index[-1]

    latest_candidates = pd.DataFrame({
        'price': prices.loc[latest_date],
        'base_score': base_score.loc[latest_date],
        'long_score': long_score.loc[latest_date],
        'shock_score': shock_score.loc[latest_date],
        'shock_return': shock_return.loc[latest_date],
        'final_score': final_score.loc[latest_date],
        'score_percentile': score_percentile.loc[latest_date],
        'breakout': breakout.loc[latest_date],
        'eligible': eligible.loc[latest_date],
        'entry_weight': entry_weights.loc[latest_date]
    })

    latest_candidates = latest_candidates[latest_candidates['eligible'] == True]
    latest_candidates = latest_candidates.sort_values('final_score', ascending=False)

    print("\n=== Current Eligible Candidates ===")
    if len(latest_candidates) == 0:
        print("No eligible candidates on the latest date")
    else:
        print(latest_candidates.head(20))

    recent = pd.DataFrame({
        'strategy_returns': strategy_returns,
        'exposure': exposure,
        'signals': signal_count
    }).tail(20)

    print("\n=== Recent Days ===")
    print(recent)

    return {
        'strategy_returns': strategy_returns,
        'exposure': exposure,
        'positions': positions,
        'entry_weights': entry_weights,
        'eligible': eligible,
        'selected': selected,
        'final_score': final_score,
        'shock_score': shock_score,
        'base_score': base_score,
        'long_score': long_score,
        'score_percentile': score_percentile,
        'strategy_stats': strategy_stats,
        'spy_stats': spy_stats,
        'universe_stats': universe_stats,
        'matched_spy_stats': matched_spy_stats,
        'matched_universe_stats': matched_universe_stats,
        'matched_spy_returns': matched_spy_returns,
        'matched_universe_returns': matched_universe_returns
    }


results = run_opportunistic_backtest()
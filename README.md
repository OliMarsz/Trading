<h1 align="center">Trading Backtests</h1>

<p align="center">
  Momentum and relative strength strategy research in Python.
</p>

<p align="center">
  <strong>Current best:</strong> Relative Strength Momentum with Market-Only Exits
</p>

---

## Performance Snapshot

| Metric                | Current Best Version |
|-----------------------|---------------------:|
| Total Return          |              99.74% |
| CAGR                  |              10.41% |
| Annual Volatility     |              17.85% |
| Sharpe Ratio          |                0.53 |
| Sortino Ratio         |                0.56 |
| Max Drawdown          |             -21.07% |
| Calmar Ratio          |                0.49 |
| Win Rate              |              74.90% |
| Average Exposure      |              46.46% |

---

## Strategy Leaderboard

| Version                         | Total Return | Max Drawdown | Avg Exposure | Status                 |
|---------------------------------|-------------:|-------------:|-------------:|------------------------|
| Market-only exit relative strength |      99.74% |      -21.07% |       46.46% | Current best overall   |
| Fixed 10-name relative strength |      58.75% |      -25.25% |       45.34% | Previous best return   |
| 12-name balanced version        |      45.28% |      -12.66% |       28.96% | Best defensive version |
| 20-name safer version           |      33.44% |      -17.30% |       44.18% | Too diluted            |
| Shock momentum versions         |          --  |          --  |          --  | Mostly abandoned       |

---

## Benchmark Comparison

| Strategy                    | Total Return |   CAGR | Max Drawdown | Avg Exposure |
|-----------------------------|-------------:|-------:|-------------:|-------------:|
| Relative Strength Momentum  |      99.74% | 10.41% |      -21.07% |       46.46% |
| SPY Buy and Hold            |     145.95% | 13.74% |      -33.72% |      100.00% |
| Equal Weight Universe       |     257.79% | 20.01% |      -33.49% |      100.00% |
| Exposure-Matched SPY        |      68.42% |  7.75% |      -13.06% |       46.46% |
| Exposure-Matched Universe   |      89.14% |  9.55% |      -12.15% |       46.46% |

The strategy does not beat SPY or the equal-weight universe on raw return. That is not the main benchmark because the bot is only invested about 46% of the time on average. The more relevant comparison is against exposure-matched benchmarks.

---

## Return Breakdown

| Component                         | Result  |
|-----------------------------------|--------:|
| Total Strategy Return             |  99.74% |
| Compounded Stock Contribution     |  91.41% |
| Compounded Cash Contribution      |   7.77% |
| Trading Cost Drag                 |   3.22% |

Most of the return came from selected stock positions, not from sitting in cash. The stock and cash contribution figures are compounded separately, while the trading cost drag is the summed transaction-cost impact across the backtest.

---

## Current Strategy Settings

| Parameter                 | Value              |
|---------------------------|--------------------|
| Strategy Type             | Relative Strength  |
| Exit Type                 | Market-only exit   |
| Universe Size             | 207 stocks         |
| Rebalance Interval        | 42 trading days    |
| Max Positions             | 12                 |
| Max Position Size         | 7%                 |
| Max Gross Exposure        | 84%                |
| Minimum Score Percentile  | 90th percentile    |
| Transaction Cost          | 0.10% per turnover |
| Backtest Period           | 2018 to 2025       |

---

## Strategy Summary

The current strategy ranks stocks by medium-term relative strength. It calculates momentum over three windows:

| Signal        | Window           | Weight |
|---------------|------------------|-------:|
| Fast momentum | 63 trading days  |  25.0% |
| Mid momentum  | 126 trading days |  45.0% |
| Slow momentum | 252 trading days |  30.0% |

Final score:

```python
relative_strength_score = (
    0.25 * rank_fast +
    0.45 * rank_mid +
    0.30 * rank_slow
)
```

The strategy selects the strongest eligible stocks on rebalance days, subject to liquidity, trend, and market regime filters.

---

## Core Idea

The main research finding was that short-term shock momentum was weak, while medium-term relative strength worked better.

The strongest improvement came from removing individual stock daily exits. Earlier versions sold individual stocks when they dipped below moving-average filters. That hurt performance because the relative strength signal appears to work over weeks and months rather than days.

The current best version holds selected names until the next rebalance unless the broader market regime breaks.

---

## Project Structure

```text
TRADING/
|
├── archive/
|   ├── opportunity_BT.py
|   └── rel_str_BT.py
|
└── Trading-main/
    ├── MTbacktesting.py
    ├── MTbot.py
    ├── README.md
    └── rel_str_ex_BT.py
```

---

## File Notes

| File                            | Purpose                                      |
|---------------------------------|----------------------------------------------|
| `Trading-main/MTbot.py`         | Original single-stock momentum bot           |
| `Trading-main/MTbacktesting.py` | Earlier backtesting / optimization script    |
| `Trading-main/rel_str_ex_BT.py` | Current best relative strength backtest      |
| `archive/opportunity_BT.py`     | Archived shock/opportunistic momentum test   |
| `archive/rel_str_BT.py`         | Archived earlier relative strength version   |

---

## Run

From the repo folder:

```bash
cd Trading-main
python rel_str_ex_BT.py
```

Dependencies:

```bash
pip install numpy pandas yfinance
```

---

## Output

The current script prints:

| Category              | Examples                                            |
|-----------------------|-----------------------------------------------------|
| Strategy stats        | return, CAGR, volatility, Sharpe, drawdown          |
| Benchmarks            | SPY, equal-weight universe, exposure-matched tests  |
| Return breakdown      | stock contribution, cash contribution, costs        |
| Trading activity      | exposure, rebalance days, number of selected names  |
| Signal diagnostics    | forward returns after selected signals              |
| Current portfolio     | top-ranked names and current holdings               |

---

## Limitations

This is a research backtester, not a live trading system.

| Limitation                  | Notes                                      |
|-----------------------------|--------------------------------------------|
| Manual universe             | Survivorship bias may be present           |
| Yahoo Finance data          | Data gaps or adjustment issues are possible |
| Simplified costs            | Slippage and spreads are not fully modeled |
| Backtest iteration          | Parameters were developed through testing  |
| No live execution           | No paper trading or broker integration yet |
| Taxes                       | Not modeled                                |

---

## Possible Next Steps

| Improvement                  | Purpose                                  |
|-----------------------------|------------------------------------------|
| Equity curve plots           | Visualize return path                    |
| Drawdown plots               | Better risk analysis                     |
| Save results to CSV          | Compare versions more easily            |
| Walk-forward validation      | Reduce overfitting risk                  |
| Sector exposure tracking     | Check concentration risk                 |
| Lower exposure cap tests     | Try to reduce drawdown                   |
| Compare against QQQ / MTUM   | Add better benchmark context             |

---

## Current Status

The current best version is profitable over the tested period and improves meaningfully on earlier versions.

The main takeaway so far:

> Medium-term relative strength worked better than short-term shock momentum, and market-only exits worked better than individual stock daily exits.

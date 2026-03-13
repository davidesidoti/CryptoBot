# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the bot

```bash
# Activate venv first
source .venv/Scripts/activate   # Windows bash
# or
.venv\Scripts\activate.bat      # Windows cmd

# Run the full pipeline (fetch → features → walk-forward → backtest)
python cryptobot.py

# To enable live trading on Binance Testnet, uncomment the last line in __main__:
# run_bot(model)
```

## Environment setup

Copy `.env.example` to `.env` and fill in Binance Testnet credentials:
```
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_SECRET=...
```
Keys are obtained from https://testnet.binance.vision/ (GitHub login required).

## Architecture

Single-file bot (`cryptobot.py`). The pipeline runs sequentially:

```
fetch_ohlcv()            → downloads 5000 OHLCV candles (paginated) from Binance public API
build_features()         → adds 20 features (1h + 4h + 1d multi-timeframe) + dynamic ATR target
walk_forward_train()     → binary XGBClassifier (BUY vs NO-BUY), 17 sliding folds
backtest()               → backtesting.py, $100k cash, 0.1% commission, P&L in $
run_bot()                → live loop on Binance Testnet every SLEEP_SECONDS
```

The model is binary: BUY (1) vs NO-BUY (0). `predict_proba` for BUY must exceed `MIN_PROBA` (0.60) to place a live order. SELL signals come from technical rules (RSI, MACD, EMA), not from the ML model.

## Key constraints

- `shuffle=False` in train/test split is mandatory — time-series data, order matters.
- `backtesting.py` requires capitalized column names (`Open`, `High`, `Low`, `Close`, `Volume`).
- Live data fetches from **Binance public API** (real prices); orders go to **Binance Testnet** (fake funds).
- The model is retrained from scratch on every run — no model persistence.
- Bot is **LONG-ONLY** on Binance Spot — no shorting.
- Invoke `trading-safety-reviewer` agent before modifying `run_bot()`.

## Features used by the model (20 total)

**1h base**: `rsi`, `macd`, `macd_signal`, `bb_width`, `vol_change`, `price_change`, `ema_cross`, `atr`, `obv_change`, `stoch_k`, `rsi_slope`, `hour`
**Multi-timeframe**: `rsi_4h`, `macd_4h`, `ema_cross_4h`, `trend_4h`, `rsi_1d`, `trend_1d`
**Regime**: `atr_ratio`, `vol_regime`

Defined in the `FEATURES` list at the top of `cryptobot.py` — adding a feature requires updating both `build_features()` and this list.

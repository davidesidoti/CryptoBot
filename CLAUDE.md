# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the bot

```bash
# Activate venv first
source .venv/Scripts/activate   # Windows bash
# or
.venv\Scripts\activate.bat      # Windows cmd

# Run the full pipeline (fetch → features → train → backtest)
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
fetch_ohlcv()        → downloads 500 OHLCV candles from Binance public API (no auth)
build_features()     → adds 7 technical indicators + supervised label (-1/0/+1)
train_model()        → XGBClassifier, 80/20 time-series split (shuffle=False)
backtest()           → backtesting.py, $10k cash, 0.1% commission
run_bot()            → live loop on Binance Testnet every SLEEP_SECONDS
```

The XGBoost model uses classes 0/1/2 internally (shifted from -1/0/+1). `predict_proba` confidence must exceed `MIN_PROBA` (default 0.50) to place a live order.

## Key constraints

- `shuffle=False` in train/test split is mandatory — time-series data, order matters.
- `backtesting.py` requires capitalized column names (`Open`, `High`, `Low`, `Close`, `Volume`).
- Live data fetches from **Binance public API** (real prices); orders go to **Binance Testnet** (fake funds).
- The model is retrained from scratch on every run — no model persistence.

## Features used by the model

`rsi`, `macd`, `macd_signal`, `bb_width`, `vol_change`, `price_change`, `ema_cross`

Defined in the `FEATURES` list at the top of `cryptobot.py` — adding a feature requires updating both `build_features()` and this list.

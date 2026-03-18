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
Keys are obtained from https://demo.binance.com (account Binance reale richiesto).
The old `testnet.binance.vision` is deprecated — ccxt uses `exchange.enable_demo_trading(True)`.

## Architecture

Single-file bot (`cryptobot.py`). The pipeline runs sequentially:

```
fetch_ohlcv()            → downloads 10000 OHLCV candles (paginated, ~416 days) from Binance public API
build_features()         → adds 23 features (1h + 4h + 1d multi-timeframe) + dynamic ATR target
optimize_hyperparams()   → Optuna bayesian search (50 trials), cached in best_params.json for 48h
walk_forward_train()     → binary XGBClassifier (BUY vs NO-BUY), ~41 sliding folds, uses Optuna params
backtest()               → backtesting.py, INITIAL_CASH (default $500), 0.1% commission, P&L in $
run_bot()                → live loop on Binance Testnet every SLEEP_SECONDS
```

The model is binary: BUY (1) vs NO-BUY (0). `predict_proba` for BUY must exceed `MIN_PROBA` (0.55) to place a live order. SELL signals come from technical rules (RSI > 75, MACD 2-bar confirmed cross, EMA20 break > 0.8%), not from the ML model. `MIN_HOLD_BARS = 5` suppresses SELL for 5h after BUY (stop loss always fires).

## Key constraints

- `shuffle=False` in train/test split is mandatory — time-series data, order matters.
- `backtesting.py` requires capitalized column names (`Open`, `High`, `Low`, `Close`, `Volume`).
- Live data fetches from **Binance public API** (real prices); orders go to **Binance Testnet** (fake funds).
- Model is persisted to `model.joblib`; delete it to force a full retrain: `rm model.joblib`
- Bot is **LONG-ONLY** on Binance Spot — no shorting.
- Invoke `trading-safety-reviewer` agent before modifying `run_bot()`.

## Gotchas

- **backtesting.py integer sizing**: with small capital vs high asset price (e.g. $500 vs BTC at $80K),
  position sizes round to 0 and no trades execute. `backtest()` auto-scales prices via `price_divisor`;
  P&L and % returns remain correct. Do not remove this logic.
- **`stats["Commissions [$]"]` missing**: absent from backtesting.py stats when `# Trades == 0`.
  Always gate with `if n_trades > 0` before accessing trade-level keys.
- **Telegram HTML mode**: use `&amp;` not `&` (e.g. `P&amp;L`) — bare `&` causes silent drop.
- **Binance Demo Trading keys**: keys from `demo.binance.com` and keys from `testnet.binance.vision`
  are not interchangeable — they are separate systems.
- **Optuna cache**: `best_params.json` has 48h TTL; delete to force re-optimization.
- **Dashboard**: `dashboard.py` (Flask on port 5050) reads `dashboard_data.json` and `price_history.json` written by bot each cycle. Run as separate process. Endpoints: `/api/status`, `/api/trades`, `/api/equity`, `/api/candles`.
- **Degenerate folds**: now use previous fold's model as fallback instead of zeroing predictions.
- **`save_state()` entry_time**: old state files without `entry_time` are handled gracefully (sell not blocked).
- **Retry con backoff**: errori di rete transitori (`ccxt.NetworkError`) vengono ritentati fino a 3 volte
  (30s, 60s, 120s) prima di notificare su Telegram. Errori non di rete notificano subito.
- **State persistence after order**: `entry_price`/`save_state()` vengono chiamati subito dopo
  `create_order()`, prima di Telegram/dashboard, per evitare posizioni "fantasma" in caso di errore.

## Features used by the model (23 total)

**1h base**: `rsi`, `macd`, `macd_signal`, `bb_width`, `vol_change`, `price_change`, `ema_cross`, `atr`, `obv_change`, `stoch_k`, `rsi_slope`, `hour`, `adx`, `willr`, `vwap_dist`
**Multi-timeframe**: `rsi_4h`, `macd_4h`, `ema_cross_4h`, `trend_4h`, `rsi_1d`, `adx_1d`
**Regime**: `atr_ratio`, `vol_regime`

Defined in the `FEATURES` list at the top of `cryptobot.py` — adding a feature requires updating both `build_features()` and this list.

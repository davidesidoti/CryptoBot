# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the bot

```bash
# Activate venv first
source .venv/Scripts/activate   # Windows bash
# or
.venv\Scripts\activate.bat      # Windows cmd

# Run the full pipeline (fetch → features → Optuna → walk-forward → backtest)
python cryptobot.py

# Run backtest only (skip training, use cached models + Optuna params)
python cryptobot.py --backtest

# To enable live trading on Binance Demo, uncomment the last line in __main__:
# run_bot(model_buy, model_short)
```

## Environment setup

Copy `.env.example` to `.env` and fill in Binance Demo Trading credentials:
```
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_SECRET=...
```
Keys are obtained from https://demo.binance.com (account Binance reale richiesto).
Same keys work for both Spot (LONG) and Futures (SHORT) — "Enable Futures" must be checked.
The old `testnet.binance.vision` is deprecated — ccxt uses `exchange.enable_demo_trading(True)`.

## Architecture

Single-file bot (`cryptobot.py`). **Dual-model LONG+SHORT** pipeline:

```
fetch_ohlcv()            → downloads 10000 OHLCV candles (paginated, ~416 days) from Binance API
build_features()         → adds 30 features (1h + 4h + 1d + 15m multi-timeframe) + dynamic ATR targets
optimize_hyperparams()   → 2x Optuna bayesian search (50 trials each: BUY + SHORT), cached 48h
walk_forward_train()     → dual XGBClassifier (BUY + SHORT), ~41 sliding folds, Optuna params
backtest()               → backtesting.py LONG+SHORT, INITIAL_CASH $500, 0.1% commission, P&L in $
run_bot()                → live loop: LONG on Spot Demo + SHORT on Futures Demo, every 15 min
```

**Two separate ML models:**
- **BUY model**: BUY (1) vs NO-BUY (0). `predict_proba >= MIN_PROBA (0.55)` to open LONG.
- **SHORT model**: SHORT (-1) vs NO-SHORT (0). `predict_proba >= SHORT_MIN_PROBA (0.65)` to open SHORT.

**Signal logic:**
- LONG close: technical rules (RSI > 75, MACD 2-bar cross, EMA20 break > 0.8%)
- SHORT close: technical cover rules (RSI < 30, MACD bullish cross, price > EMA20 + 0.5%)
- `MIN_HOLD_BARS = 5` suppresses LONG close for 5h; `SHORT_MIN_HOLD = 3` for 3h
- Trailing stop (ATR-based): initial stop = max(entry - ATR×2, entry - 2%), trails up with price
- Entry filters: LONG requires `mom_15m > 0`; SHORT requires `mom_15m < 0` AND `ADX > 20`
- Conflict: if both BUY and SHORT fire simultaneously → HOLD (do nothing)

## Key constraints

- `shuffle=False` in train/test split is mandatory — time-series data, order matters.
- `backtesting.py` requires capitalized column names (`Open`, `High`, `Low`, `Close`, `Volume`).
- Live data from **Binance public API** (real prices, cached exchange instance); orders to **Demo Trading**.
- LONG orders go to **Spot Demo** (`demo-api.binance.com`); SHORT orders go to **Futures Demo** (`demo-fapi.binance.com`).
- Models persisted as `model_buy.joblib` + `model_short.joblib`; delete to force retrain.
- Futures leverage forced to **1x** (no leverage). `set_leverage(1)` fails → bot refuses to start.
- Invoke `trading-safety-reviewer` agent before modifying `run_bot()`.
- Every `create_order()` is wrapped in try/except — state only updates after confirmed execution.
- `position_type` ("long"/"short"/None) in `bot_state.json` — critical for correct stop-loss direction after restart.

## Gotchas

- **backtesting.py integer sizing**: with small capital vs high asset price (e.g. $500 vs BTC at $80K),
  position sizes round to 0 and no trades execute. `backtest()` auto-scales prices via `price_divisor`;
  P&L and % returns remain correct. Do not remove this logic.
- **`stats["Commissions [$]"]` missing**: absent from backtesting.py stats when `# Trades == 0`.
  Always gate with `if n_trades > 0` before accessing trade-level keys.
- **Telegram HTML mode**: use `&amp;` not `&` (e.g. `P&amp;L`) — bare `&` causes silent drop.
- **Binance Demo Trading keys**: keys from `demo.binance.com` and keys from `testnet.binance.vision`
  are not interchangeable — they are separate systems.
- **Optuna cache**: `best_params_buy.json` and `best_params_short.json` have 48h TTL; delete to force re-optimization.
- **Dashboard**: `dashboard.py` (Flask on port 5050) reads `dashboard_data.json` and `price_history.json`.
  Includes signal log with action/reason for each cycle. Endpoints: `/api/status`, `/api/trades`, `/api/equity`, `/api/candles`.
- **Degenerate folds**: use previous fold's model as fallback instead of zeroing predictions.
- **`save_state()` position_type + trail_stop**: old state files without `position_type` default to "long" for backward compat. `trail_stop` persists the current trailing stop price across restarts.
- **Retry con backoff**: transient network errors retried 3x (30s, 60s, 120s) before Telegram notification.
- **State persistence after order**: `save_state()` called immediately after `create_order()`, before Telegram/dashboard.
- **Separate balance pools**: Spot and Futures have separate USDT balances. Bot fetches from the correct exchange.
- **fetch_ohlcv caching**: exchange instance is cached globally to avoid repeated `load_markets()` calls.
  Falls back to Demo endpoint if public API (`api.binance.com`) is unreachable from VPS.
- **No double-trade per cycle**: after any stop-loss exit, `continue` prevents opening new position same cycle.
- **P&L direction**: `_calc_pnl()` helper handles LONG (price - entry) and SHORT (entry - price) correctly.

## Features used by both models (30 total)

**1h base**: `rsi`, `macd`, `macd_signal`, `macd_hist`, `bb_width`, `vol_change`, `price_change`, `ema_cross`, `atr`, `obv_change`, `stoch_k`, `rsi_slope`, `hour`, `adx`, `willr`, `vwap_dist`
**Multi-timeframe**: `rsi_4h`, `macd_4h`, `ema_cross_4h`, `trend_4h`, `rsi_1d`, `adx_1d`
**Regime**: `atr_ratio`, `vol_regime`
**Strategy**: `trend_down` (price < EMA20, used as feature + SHORT trend filter)
**15m sub-hourly** (early entry): `rsi_15m`, `macd_15m`, `macd_hist_15m`, `mom_15m`, `vol_spike_15m`

15m features are fetched separately (can't resample up from 1h). `build_features()` auto-fetches 15m data if not provided; `run_bot()` passes fresh 15m candles each 15-min cycle for up-to-date sub-hourly signals.

Defined in the `FEATURES` list at the top of `cryptobot.py` — adding a feature requires updating both `build_features()` and this list.

## Gotchas (deployment / debug)

- **VPS crash loop diagnosis**: if bot loads model then exits without "Bot avviato", check if `run_bot()` is
  uncommented. If traceback shows jinja2/typing error with `KeyboardInterrupt`, it's systemd stopping during
  import — not a real jinja2 bug. Check `journalctl -u cryptobot -n 100 --no-pager` for full context.
- **VPS Python 3.13**: Hostinger VPS runs Python 3.13 which can have compatibility issues with older
  versions of bokeh/jinja2. Keep `pip install --upgrade jinja2 markupsafe bokeh` in deployment steps.
- **`save_dashboard_data()` signature**: takes `short_proba`, `position_type`, `action_taken`, `reason`
  as optional kwargs. Signal log accumulates last 50 entries in `dashboard_data.json["signal_log"]`.

## Quick validation

```bash
# Syntax check (no execution)
python -c "import py_compile; py_compile.compile('cryptobot.py', doraise=True)"

# Test model load + state functions
python -c "from cryptobot import load_model, load_state; load_model(); load_state()"
```

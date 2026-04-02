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
**"Enable Futures" must be checked** — this scalping version routes ALL orders through Futures.
The old `testnet.binance.vision` is deprecated — ccxt uses `exchange.enable_demo_trading(True)`.

## Architecture

Single-file scalping bot (`cryptobot.py`). **Dual-model LONG+SHORT** pipeline on **5m candles**:

```
fetch_ohlcv()            → downloads 10000 OHLCV 5m candles (paginated, ~34 days) from Binance API
build_features()         → adds 28 scalping features (5m + 15m + 1h multi-timeframe) + dynamic ATR targets
optimize_hyperparams()   → 2x Optuna bayesian search (60 trials each: BUY + SHORT), cached 48h
walk_forward_train()     → dual XGBClassifier (BUY + SHORT), ~20 sliding folds, Optuna params
backtest()               → backtesting.py LONG+SHORT, INITIAL_CASH $500, 0.04% Futures commission, P&L in $
run_bot()                → live loop: LONG + SHORT on Futures Demo, every 60s (ML signals every 5m)
```

**Two separate ML models:**
- **BUY model**: BUY (1) vs NO-BUY (0). `predict_proba >= MIN_PROBA (0.58)` to open LONG.
- **SHORT model**: SHORT (-1) vs NO-SHORT (0). `predict_proba >= SHORT_MIN_PROBA (0.60)` to open SHORT.

**Signal logic:**
- LONG close: take profit (0.6%), trailing stop (ATR×1.5 after +0.3%), stop loss (0.4%), technical rules (RSI>70, MACD cross, EMA20 break)
- SHORT close: take profit (0.6%), trailing stop, stop loss (0.4%), technical cover rules (RSI<35, MACD bullish, EMA20 break)
- `MIN_HOLD_BARS = 3` suppresses close for 15 min (3 bars × 5m) for LONG; `SHORT_MIN_HOLD = 3` (separate constant, line 80) for SHORT
- Entry filters: LONG requires `trend_1h == 1` AND `trend_15m == 1`; SHORT requires `trend_1h == 0` AND `adx_1h > 15`
- Conflict: if both BUY and SHORT fire simultaneously → HOLD (do nothing)

**Exit priority order:**
1. Take profit (0.6%)
2. Trailing stop (ATR×1.5, activated after +0.3%)
3. Stop loss (0.4%)
4. Technical close signals
5. Hold minimum (15 min)

## Key constraints

- `shuffle=False` in train/test split is mandatory — time-series data, order matters.
- `backtesting.py` requires capitalized column names (`Open`, `High`, `Low`, `Close`, `Volume`).
- Live data from **Binance public API** (real prices); orders to **Futures Demo** (`demo-fapi.binance.com`).
- **Futures-only**: both LONG and SHORT go through Futures Demo for 0.02% maker fee (vs 0.1% spot).
- Models persisted as `model_buy_scalp.joblib` + `model_short_scalp.joblib`; delete to force retrain.
- Futures leverage forced to **1x** (no leverage). `set_leverage(1)` fails → bot refuses to start.
- Invoke `trading-safety-reviewer` agent before modifying `run_bot()`.
- Every `create_order()` is wrapped in try/except — state only updates after confirmed execution.
- `position_type` ("long"/"short"/None) in `bot_state.json` — critical for correct stop-loss direction after restart.
- **Bot cycle**: checks TP/SL/trailing every 60s, generates ML signals only on 5m candle close (`minute % 5 == 0`).

## Gotchas

- **backtesting.py integer sizing**: with small capital vs high asset price (e.g. $500 vs BTC at $80K),
  position sizes round to 0 and no trades execute. `backtest()` auto-scales prices via `price_divisor`;
  P&L and % returns remain correct. Do not remove this logic.
- **`stats["Commissions [$]"]` missing**: absent from backtesting.py stats when `# Trades == 0`.
  Always gate with `if n_trades > 0` before accessing trade-level keys.
- **Telegram HTML mode**: use `&amp;` not `&` (e.g. `P&amp;L`) — bare `&` causes silent drop.
- **Binance Demo Trading keys**: keys from `demo.binance.com` and keys from `testnet.binance.vision`
  are not interchangeable — they are separate systems.
- **Optuna cache**: `best_params_buy_scalp.json` and `best_params_short_scalp.json` have 48h TTL; delete to force re-optimization.
- **Dashboard**: `dashboard.py` (Flask on port 5050) reads `dashboard_data.json` and `price_history.json`.
  Includes signal log with action/reason for each cycle. Endpoints: `/api/status`, `/api/trades`, `/api/equity`, `/api/candles`.
- **Degenerate folds**: use previous fold's model as fallback instead of zeroing predictions.
- **`save_state()` / `load_state()`**: returns 10-tuple `(entry_price, qty, entry_time, position_type, trail_stop, consecutive_sl, daily_pnl, daily_reset_date, pause_until, cooldown_until)`. All new fields use `.get(key, default)` for backward compat with old state files.
- **Circuit breaker**: 2 consecutive stop-losses → `pause_until = now + 2h`; `daily_pnl / initial_capital ≤ -1.5%` → `pause_until = midnight UTC`. Win/TP resets `consecutive_sl`. State persists across restarts via `bot_state.json`.
- **Cooldown post-uscita**: after SELL_TECH/COVER_TECH → `cooldown_until = now + 30min` (loss) or `+5min` (win); after TP/trailing → `+5min`; after SL → no cooldown (CB handles it). Blocks new entries only, never exits. Persists via `bot_state.json`.
- **`SHORT_MIN_HOLD` is separate from `MIN_HOLD_BARS`** (line 80) — both must be updated when changing minimum hold for both directions.
- **Retry con backoff**: transient network errors retried 3x (5s, 15s, 30s) before Telegram notification. Faster than swing version because scalping needs quick recovery.
- **State persistence after order**: `save_state()` called immediately after `create_order()`, before Telegram/dashboard.
- **fetch_ohlcv caching**: exchange instance is cached globally to avoid repeated `load_markets()` calls.
- **No double-trade per cycle**: after any stop-loss/TP exit, `continue` prevents opening new position same cycle.
- **P&L direction**: `_calc_pnl()` helper handles LONG (price - entry) and SHORT (entry - price) correctly.
- **5m data = ~34 days**: with FETCH_LIMIT=10000 and 5m candles, the dataset covers only ~34 days. Walk-forward yields ~20 folds instead of ~41.
- **Take profit before trailing**: TP check runs before trailing stop in `next()` and `run_bot()`. If both trigger on same bar, TP wins.
- **Futures-only mode**: `USE_FUTURES_FOR_BOTH=True` routes all orders through Futures. Disabling it falls back to Spot+Futures like the main branch.
- **15m and 1h features are resampled from 5m**: no separate fetch needed. `build_features()` takes only one df parameter (5m OHLCV).
- **EnsembleClassifier (max-with-agreement)**: instead of mean, returns max of 5 models when ≥3 agree.
  BUY uses `agree_thresh=0.50`; SHORT uses `agree_thresh=0.40` (SHORT signals are less pronounced individually).
  Calibration caps threshold at `min(calibrated, max(default, P90_distribution))` — prevents impossible thresholds.
  If startup log shows "BUY=58%/SHORT=60%" (defaults), it means P90 cap kicked in — normal, not failure.
  Look at `[LIVE-DIAG]` to see actual proba vs threshold on each 5m candle.
- **Calibration uses only last ENSEMBLE_SIZE folds** (not all OOS data) to avoid look-ahead bias.
  Evaluating ensemble on early-fold OOS data inflates probabilities (later-trained models see "future").
- **Backtest uses per-fold models, not ensemble**: OOS backtest shows more trades than live trading.
  Comparing backtest trade count vs live is misleading — ensemble threshold is higher than per-fold.

## Features used by both models (28 total)

**5m base oscillators/momentum (11)**: `rsi_fast`, `rsi_slope`, `macd_fast`, `macd_signal_fast`, `macd_hist_fast`, `stoch_k`, `stoch_d`, `bb_width`, `bb_pct`, `willr`, `cci`
**5m price/volume dynamics (7)**: `price_change`, `price_change_3`, `vol_change`, `vol_spike`, `vwap_dist`, `atr`, `atr_ratio`
**5m microstructure (3)**: `spread_proxy`, `body_ratio`, `ema_cross_fast`
**15m context (resampled from 5m) (3)**: `rsi_15m`, `macd_15m`, `trend_15m`
**1h context (resampled from 5m) (3)**: `rsi_1h`, `trend_1h`, `adx_1h`
**Time (1)**: `minute_of_day`

Defined in the `FEATURES` list at the top of `cryptobot.py` — adding a feature requires updating both `build_features()` and this list.

## Gotchas (deployment / debug)

- **VPS crash loop diagnosis**: if bot loads model then exits without "Bot avviato", check if `run_bot()` is
  uncommented. If traceback shows jinja2/typing error with `KeyboardInterrupt`, it's systemd stopping during
  import — not a real jinja2 bug. Check `journalctl -u cryptobot -n 100 --no-pager` for full context.
- **VPS Python 3.13**: Hostinger VPS runs Python 3.13 which can have compatibility issues with older
  versions of bokeh/jinja2. Keep `pip install --upgrade jinja2 markupsafe bokeh` in deployment steps.
- **`save_dashboard_data()` signature**: takes `short_proba`, `position_type`, `action_taken`, `reason`,
  `eff_buy_thresh`, `eff_short_thresh` as optional kwargs. Pass calibrated thresholds (not MIN_PROBA constants)
  so dashboard shows actual effective thresholds. Signal log accumulates last 50 entries.
- **Commission impact for scalping**: Futures 0.04% round-trip on a 0.5% target = ~8% of profit. Spot 0.2% round-trip = ~40% of profit. Always use Futures for scalping.
- **Live vs backtest trade drought diagnosis**: put `dashboard_data.json` and `price_history.json` from
  VPS into `debugging/` folder. Check `signal_log[n]["reason"]` — shows effective threshold and proba.
  If "Confidenza bassa (BUY X% < 58%)" consistently, calibration failed or model not retrained yet.
  Delete `.joblib` files to force retrain with new calibration.

## Quick validation

```bash
# Syntax check (no execution)
python -c "import py_compile; py_compile.compile('cryptobot.py', doraise=True)"

# Test model load + state functions
python -c "from cryptobot import load_model, load_state; load_model(); load_state()"
```

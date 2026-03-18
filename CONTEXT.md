# cryptobot — Project Context for Claude Code

## Obiettivo del progetto

Bot di trading crypto basato su ML che opera su **Binance Demo Trading** (soldi finti, dati reali).
Dual-model: **LONG** (Spot) + **SHORT** (Futures), entrambi su demo endpoint.
Il bot scarica candele OHLCV, genera feature tecniche, ottimizza iperparametri con Optuna,
allena due classificatori XGBoost con walk-forward validation, backtesta la strategia LONG+SHORT,
e gira in loop live piazzando ordini su Binance Demo.
Dashboard web (Flask) per monitoraggio real-time con signal log dettagliato.

---

## Stack tecnico

| Componente          | Libreria / Servizio               |
|---------------------|-----------------------------------|
| Dati di mercato     | `ccxt` + Binance public API       |
| Feature engineering | `pandas-ta`                       |
| Modello ML          | `XGBClassifier` (xgboost) x2      |
| Ottimizzazione HP   | `optuna` (2x: BUY + SHORT)       |
| Backtest            | `backtesting.py`                  |
| Esecuzione LONG     | `ccxt` + Binance Spot Demo        |
| Esecuzione SHORT    | `ccxt` + Binance Futures Demo     |
| Dashboard           | `flask` + Chart.js + Tailwind CSS |
| Notifiche           | Telegram Bot API                  |
| Linguaggio          | Python 3.10+                      |

### Install

```bash
pip install -r requirements.txt
# oppure:
pip install ccxt pandas numpy scikit-learn xgboost pandas-ta backtesting flask optuna
```

---

## Architettura del bot

```
Binance public API (dati OHLCV storici, 10000 candele paginato, ~416 giorni)
        |
build_features() -> 25 feature (1h + 4h + 1d multi-timeframe) + target dinamico ATR
        |
optimize_hyperparams() -> 2x Optuna bayesian search (50 trial ciascuno), cache 48h
        |
walk_forward_train() -> dual XGBoost (BUY + SHORT), ~41 fold scorrevoli
        |
backtest() -> backtesting.py LONG+SHORT -> stats + P&L in $ + plot
        |
run_bot() -> loop live: LONG su Spot Demo + SHORT su Futures Demo, ogni 15 min
```

---

## File del progetto

### File principali
- **`cryptobot.py`** — bot completo, tutto autocontenuto (dual model LONG+SHORT)
- **`dashboard.py`** — Flask web app per monitoraggio real-time (porta 5050)
- **`templates/dashboard.html`** — dashboard HTML (dark theme, Chart.js, signal log, auto-refresh 30s)

### Sezioni principali di `cryptobot.py`

1. **CONFIG** (in cima al file) — tutte le variabili configurabili (incluse SHORT_*)
2. **`fetch_ohlcv()`** — scarica candele da Binance con paginazione (exchange instance cached, fallback demo endpoint)
3. **`build_features()`** — indicatori 1h + multi-timeframe (4h, 1d) + target dinamico ATR per BUY e SHORT
4. **`optimize_hyperparams()`** — Optuna bayesian search, separata per BUY e SHORT, cache su disco 48h
5. **`train_model()`** — XGBoost binary, usato per retraining live (target_label=1 per BUY, -1 per SHORT)
6. **`technical_sell_signal()`** — segnali chiusura LONG (RSI > 75, MACD 2-bar, EMA20 break)
7. **`technical_cover_signal()`** — segnali chiusura SHORT (RSI < 30, MACD bullish, prezzo > EMA20)
8. **`walk_forward_train()`** — walk-forward dual model, ~41 fold scorrevoli, parametri Optuna
9. **`backtest()`** — strategia ML LONG+SHORT su backtesting.py con stop loss, trend filter, hold minimo
10. **`save_dashboard_data()`** — scrive snapshot con signal_log (action, reason, trend, position_type)
11. **`get_testnet_exchange()`** — factory: "spot" per LONG, "future" per SHORT (leverage 1x, fail-fast)
12. **`save_state()` / `load_state()`** — persistenza posizione + entry_time + position_type su JSON
13. **`_calc_pnl()`** — helper P&L corretto per LONG e SHORT
14. **`run_bot()`** — loop live: dual exchange, dual model, conflict resolution, try/except su ogni ordine
15. **`__main__`** — pipeline + flag `--backtest` per skip training

---

## Variabili di configurazione (CONFIG block)

```python
SYMBOL          = "BTC/USDT"
TIMEFRAME       = "1h"
FETCH_LIMIT     = 10000       # candele storiche (~416 giorni, paginato)
FUTURE_BARS     = 5           # candele in avanti per calcolare il label
SIGNAL_THRESH   = 0.007       # soglia minima 0.7%, override da ATR dinamico
MIN_PROBA       = 0.55        # confidenza minima per ordine BUY
TRADE_SIZE      = 0.95        # % del capitale per ogni ordine LONG
STOP_LOSS       = 0.02        # 2% stop loss LONG
INITIAL_CASH    = 500         # capitale backtest (USD)
SLEEP_SECONDS   = 900         # pausa tra cicli del bot (15 min)
MIN_HOLD_BARS   = 5           # hold minimo 5h prima di CLOSE LONG

# SHORT-specific
ENABLE_SHORT      = True
SHORT_MIN_PROBA   = 0.60      # confidenza minima per ordine SHORT (piu' alta di BUY)
SHORT_TRADE_SIZE  = 0.70      # 70% del capitale per SHORT (piu' conservativo)
SHORT_STOP_LOSS   = 0.02      # 2% stop loss SHORT
SHORT_MIN_HOLD    = 3         # hold minimo 3h prima di CLOSE SHORT

# Infrastruttura
MAX_RETRIES     = 3
RETRY_BACKOFF   = [30, 60, 120]
RETRAIN_HOURS   = 24
MODEL_BUY_FILE  = "model_buy.joblib"
MODEL_SHORT_FILE = "model_short.joblib"
STATE_FILE      = "bot_state.json"
DASHBOARD_FILE  = "dashboard_data.json"
OPTUNA_BUY_FILE = "best_params_buy.json"
OPTUNA_SHORT_FILE = "best_params_short.json"
OPTUNA_TRIALS   = 50
WF_TRAIN_BARS   = 1500
WF_TEST_BARS    = 200
```

---

## Feature usate dai modelli (25 totali)

```python
FEATURES = [
    # === Timeframe 1h (base) ===
    "rsi", "macd", "macd_signal", "macd_hist", "bb_width",
    "vol_change", "price_change", "ema_cross", "atr",
    "obv_change", "stoch_k", "rsi_slope", "hour",
    "adx", "willr", "vwap_dist",
    # === Multi-timeframe (resample da 1h) ===
    "rsi_4h", "macd_4h", "ema_cross_4h", "trend_4h",
    "rsi_1d", "adx_1d",
    # === Regime / volatilita' ===
    "atr_ratio", "vol_regime",
    # === Trend direction ===
    "trend_down",   # prezzo < EMA20 (feature + filtro SHORT)
]
```

---

## Label / target (dual)

**BUY model** — Binary classification: BUY (1) vs NO-BUY (0).
- `1` (BUY) -> prezzo sale > soglia dinamica nelle prossime 5 candele
- Soglia dinamica: `max(0.7%, ATR% * 0.5)`

**SHORT model** — Binary classification: SHORT (-1) vs NO-SHORT (0).
- `-1` (SHORT) -> prezzo scende > soglia dinamica nelle prossime 5 candele
- Stessa soglia dinamica ATR

**Chiusura posizioni** (regole tecniche, non ML):
- CLOSE LONG: RSI > 75 in discesa, MACD cross ribassista 2-bar, prezzo < EMA20 drop > 0.8%
- CLOSE SHORT: RSI < 30 in salita, MACD cross rialzista 2-bar, prezzo > EMA20 gain > 0.5%

---

## Logica ordini nel bot live (LONG + SHORT)

```
# Stop loss (SEMPRE attivo, prima di tutto)
if posizione LONG and loss >= 2%: chiudi su Spot, continue (no nuova posizione)
if posizione SHORT and loss >= 2%: chiudi su Futures, continue

# Conflict resolution
if buy_signal AND short_signal: entrambi a False -> HOLD

# Mutual exclusion
if buy_signal AND posizione SHORT: buy_signal = False
if short_signal AND posizione LONG: short_signal = False

# Hold minimo (sopprime chiusura tecnica, stop loss escluso)
if sell_signal and ore_hold < 5h: sell_signal = False
if cover_signal and ore_hold < 3h: cover_signal = False

# OPEN LONG (ML BUY signal)
if buy_proba >= 55% and USDT > $10 and FLAT and trend_up (prezzo > EMA20):
    BUY su Spot Demo, salva stato position_type="long"

# OPEN SHORT (ML SHORT signal)
if short_proba >= 60% and USDT > $10 and FLAT and trend_down (prezzo < EMA20):
    SELL su Futures Demo (leva 1x), salva stato position_type="short"

# CLOSE LONG (segnale tecnico)
if sell_signal and posizione LONG:
    SELL su Spot Demo, azzera stato

# CLOSE SHORT (segnale tecnico cover)
if cover_signal and posizione SHORT:
    BUY(cover) su Futures Demo, azzera stato

altrimenti: HOLD (con motivo dettagliato nel signal_log)

# Dashboard (ogni ciclo)
salva snapshot + signal_log con action/reason/trend/position_type

# Retraining periodico (ogni 24h)
riaddestra entrambi i modelli con dati freschi
```

---

## Binance Demo Trading — setup

1. Vai su **https://demo.binance.com/** (richiede account Binance reale)
2. Accedi con il tuo account Binance
3. Genera API Key dal pannello Demo Trading
4. Assicurati che **Enable Futures** sia spuntato
5. Copia API Key e Secret nel file `.env`

Stesse chiavi funzionano per Spot Demo e Futures Demo.
- Spot Demo endpoint: `demo-api.binance.com`
- Futures Demo endpoint: `demo-fapi.binance.com`

---

## Dashboard di monitoraggio

```bash
python dashboard.py
# Apri http://localhost:5050 (o http://IP_VPS:5050)
```

Dashboard web (Flask + Tailwind CSS + Chart.js + Lightweight Charts) con:
- Cards: prezzo BTC (con sparkline), segnale, confidenza ML, saldo USDT/BTC, P&L
- Countdown bar al prossimo ciclo bot
- **Decisione Corrente**: azione presa + motivo dettagliato + trend + posizione
- **Modelli ML**: barre di confidenza BUY e SHORT con soglie visualizzate
- **Signal Log**: cronologia scrollabile degli ultimi 50 cicli (orario, prezzo, trend, azione, motivo)
- Grafico candlestick BTC/USDT (ultime 100 candele 1h, TradingView Lightweight Charts)
- Equity curve interattiva
- Radar delle feature ML
- Activity feed trade recenti
- Toast notification animati per nuovi trade
- Tabella ultimi 20 trade
- Dark theme, responsive, auto-refresh 30s

---

## Stato attuale del progetto

- [x] Fetch dati OHLCV paginato (10000 candele, ~416 giorni, exchange cached + fallback demo)
- [x] Feature engineering multi-timeframe (25 feature: 1h + 4h + 1d + regime + trend)
- [x] Dual model: BUY + SHORT con ottimizzazione Optuna separata (50 trial ciascuno)
- [x] Walk-forward validation (~41 fold, fallback per fold degeneri)
- [x] Backtest LONG+SHORT con backtesting.py (stats + plot + P&L in $)
- [x] CLOSE LONG: regole tecniche (RSI 75, MACD 2-bar, EMA 0.8%)
- [x] CLOSE SHORT: regole tecniche cover (RSI 30, MACD bullish, EMA 0.5%)
- [x] Stop loss 2% per LONG e SHORT
- [x] Hold minimo: 5h LONG, 3h SHORT
- [x] Filtro trend EMA20: BUY solo in uptrend, SHORT solo in downtrend
- [x] Conflict resolution: BUY + SHORT simultanei -> HOLD
- [x] Loop live: LONG su Spot Demo, SHORT su Futures Demo (leva 1x, fail-fast)
- [x] try/except su ogni create_order(), stato aggiornato solo dopo conferma
- [x] No double-trade per ciclo (continue dopo stop-loss)
- [x] Retraining automatico ogni 24h (entrambi i modelli)
- [x] Persistenza modello su disco (model_buy.joblib + model_short.joblib)
- [x] Persistenza stato con position_type (bot_state.json, backward compat)
- [x] Notifiche Telegram (LONG/SHORT trade, stop loss, errori, retraining)
- [x] Dashboard web con signal log dettagliato (action, reason, trend)
- [x] Flag `--backtest` per skip training e usare modelli cached
- [x] Deploy su VPS (Hostinger Game Panel 4)

---

## Performance backtest (walk-forward LONG+SHORT, 342 giorni, B&H -9%)

| Metrica | Valore |
|---------|--------|
| Return | +35.0% |
| Sharpe | 1.33 |
| Sortino | 3.19 |
| Profit Factor | 1.48 |
| Win Rate | 47.5% |
| # Trade | 177 |
| Max Drawdown | -11.9% |
| Avg Trade | +0.22% |
| Avg Duration | 15h |
| SQN | 1.49 |
| Exposure Time | 34.5% |

---

## Note importanti

- Il bot usa **Binance Demo Trading** per gli ordini: nessun soldo reale viene toccato.
- I dati OHLCV vengono scaricati dalla **Binance pubblica** (mercato reale), con fallback su demo endpoint.
- Due modelli separati: `model_buy.joblib` (LONG) e `model_short.joblib` (SHORT).
- Lo stato include `position_type` ("long"/"short"/None) — critico per direzione stop-loss dopo restart.
- `shuffle=False` nel train/test split e' fondamentale per time-series.
- Commission 0.1% nel backtest, uguale a Binance.
- Optuna cache: `best_params_buy.json` e `best_params_short.json` con TTL 48h.
- Futures leverage forzato a 1x. Se `set_leverage(1)` fallisce, il bot non parte.
- Spot e Futures hanno bilanci USDT separati. Il bot interroga l'exchange corretto per ogni posizione.
- `_calc_pnl()` gestisce LONG (price - entry) e SHORT (entry - price) correttamente.
- `save_dashboard_data()` include `signal_log` con action/reason per ogni ciclo (ultimi 50).

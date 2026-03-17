# cryptobot — Project Context for Claude Code

## Obiettivo del progetto

Bot di trading crypto basato su ML che opera su **Binance Testnet** (soldi finti, dati reali).
Il bot scarica candele OHLCV, genera feature tecniche, ottimizza iperparametri con Optuna,
allena un classificatore XGBoost con walk-forward validation, backtesta la strategia,
e gira in loop live piazzando ordini sul testnet di Binance.
Dashboard web (Flask) per monitorare il bot in tempo reale.

---

## Stack tecnico

| Componente          | Libreria / Servizio               |
|---------------------|-----------------------------------|
| Dati di mercato     | `ccxt` + Binance public API       |
| Feature engineering | `pandas-ta`                       |
| Modello ML          | `XGBClassifier` (xgboost)         |
| Ottimizzazione HP   | `optuna`                          |
| Backtest            | `backtesting.py`                  |
| Esecuzione ordini   | `ccxt` + Binance Testnet          |
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
        ↓
build_features() → 23 feature (1h + 4h + 1d multi-timeframe) + target dinamico ATR
        ↓
optimize_hyperparams() → Optuna bayesian search (50 trial), cache 48h in best_params.json
        ↓
walk_forward_train() → XGBoost binary (BUY vs NO-BUY), ~41 fold scorrevoli, usa parametri Optuna
        ↓
backtest() → backtesting.py → stats + P&L in $ + plot interattivo
        ↓
run_bot() → loop live su Binance Testnet ogni SLEEP_SECONDS (15 min)
```

---

## File del progetto

### File principali
- **`cryptobot.py`** — bot completo, tutto autocontenuto
- **`dashboard.py`** — Flask web app per monitoraggio real-time (porta 5050)
- **`templates/dashboard.html`** — dashboard HTML (dark theme, Chart.js, auto-refresh 30s)

### Sezioni principali di `cryptobot.py`

1. **CONFIG** (in cima al file) — tutte le variabili configurabili
2. **`fetch_ohlcv()`** — scarica candele da Binance con paginazione (endpoint pubblico, no auth)
3. **`build_features()`** — indicatori 1h + multi-timeframe (4h, 1d) + target dinamico ATR
4. **`optimize_hyperparams()`** — Optuna bayesian search, cache su disco per 48h
5. **`train_model()`** — XGBoost binary (BUY vs NO-BUY), usato per retraining live
6. **`technical_sell_signal()`** — segnali SELL basati su regole tecniche (RSI > 75, MACD 2-bar, EMA20)
7. **`walk_forward_train()`** — walk-forward con ~41 fold scorrevoli, usa parametri Optuna
8. **`backtest()`** — strategia ML su backtesting.py con stop loss, trend filter, hold minimo, P&L in $
9. **`save_dashboard_data()`** — scrive snapshot del ciclo per la dashboard web
10. **`get_testnet_exchange()`** — istanza ccxt puntata al Binance Demo Trading
11. **`save_state()` / `load_state()`** — persistenza posizione + entry_time su JSON
12. **`log_trade()`** — logging trade su CSV
13. **`run_bot()`** — loop live con BUY ML + SELL tecnico + stop loss + hold minimo + trend filter
14. **`__main__`** — esegue fetch → ottimizzazione → walk-forward → backtest;
    `run_bot()` e' commentato, da decommentare per andare live

---

## Variabili di configurazione (CONFIG block)

```python
SYMBOL          = "BTC/USDT"
TIMEFRAME       = "1h"
FETCH_LIMIT     = 10000       # candele storiche (~416 giorni, paginato)
FUTURE_BARS     = 5           # candele in avanti per calcolare il label
SIGNAL_THRESH   = 0.007       # soglia minima 0.7%, override da ATR dinamico
MIN_PROBA       = 0.55        # confidenza minima per ordine BUY
TRADE_SIZE      = 0.95        # % del capitale per ogni ordine
STOP_LOSS       = 0.02        # 2% stop loss
INITIAL_CASH    = 500         # capitale backtest (USD)
SLEEP_SECONDS   = 900         # pausa tra cicli del bot (15 min)
MIN_HOLD_BARS   = 5           # hold minimo 5h prima di SELL tecnico (stop loss escluso)
LOG_FILE        = "trades_log.csv"
RETRAIN_HOURS   = 24          # riaddestra ogni N ore
MODEL_FILE      = "model.joblib"
STATE_FILE      = "bot_state.json"
DASHBOARD_FILE  = "dashboard_data.json"
OPTUNA_FILE     = "best_params.json"
OPTUNA_TRIALS   = 50          # trial per ricerca bayesiana
WF_TRAIN_BARS   = 1500        # walk-forward: candele per finestra training
WF_TEST_BARS    = 200         # walk-forward: candele per finestra test
```

---

## Feature usate dal modello (23 totali)

```python
FEATURES = [
    # === Timeframe 1h (base) ===
    "rsi",          # RSI 14
    "macd",         # MACD line (12,26,9)
    "macd_signal",  # MACD signal line
    "bb_width",     # Larghezza Bollinger Bands normalizzata
    "vol_change",   # Variazione % volume candela precedente
    "price_change", # Variazione % prezzo candela precedente
    "ema_cross",    # EMA9 - EMA21 (differenza)
    "atr",          # Average True Range 14
    "obv_change",   # % variazione OBV
    "stoch_k",      # Stochastic K
    "rsi_slope",    # diff(RSI, 5) — momentum dell'RSI
    "hour",         # ora della candela (pattern intraday)
    "adx",          # Average Directional Index 14 (forza trend)
    "willr",        # Williams %R 14 (ipercomprato/ipervenduto)
    "vwap_dist",    # distanza % dal VWAP rolling 20
    # === Multi-timeframe (resample da 1h) ===
    "rsi_4h",       # RSI su candele 4h
    "macd_4h",      # MACD su candele 4h
    "ema_cross_4h", # EMA9-EMA21 su candele 4h
    "trend_4h",     # prezzo > EMA20 su 4h (0/1)
    "rsi_1d",       # RSI su candele daily
    "adx_1d",       # ADX su candele daily (forza trend daily)
    # === Regime / volatilita' ===
    "atr_ratio",    # ATR(7) / ATR(28) — >1 = vol in aumento
    "vol_regime"    # volume / SMA(volume, 20) — >1 = volume sopra media
]
```

---

## Label / target

**Binary classification**: BUY (1) vs NO-BUY (0).

- `1` (BUY) → prezzo sale > soglia dinamica nelle prossime 5 candele
- `0` (NO-BUY) → tutto il resto (HOLD + SELL)

La soglia e' **dinamica**, basata su ATR: `max(0.7%, ATR% * 0.5)`.
In mercati volatili la soglia si alza automaticamente, riducendo i falsi segnali.

Il segnale SELL e' gestito da **regole tecniche** (non ML):
- RSI > 75 e in discesa (ipercomprato)
- MACD cross ribassista confermato (2 barre consecutive)
- Prezzo sotto EMA20 con drop > 0.8%

**Hold minimo**: SELL tecnico soppresso per 5h dopo BUY. Stop loss (2%) sempre attivo.

---

## Logica ordini nel bot live (solo LONG, Binance Spot)

```
# Stop loss (SEMPRE attivo, anche durante hold minimo)
if posizione_aperta and loss >= STOP_LOSS (2%):
    chiudi posizione, logga su CSV, notifica Telegram

# Hold minimo: sopprime SELL tecnico se posizione < MIN_HOLD_BARS ore
if sell_signal and ore_hold < MIN_HOLD_BARS (5h):
    sell_signal = False

# Filtro trend
if prezzo < EMA20: blocca tutti i BUY

# BUY (segnale ML binary)
if buy_proba >= MIN_PROBA (0.55) and USDT > $10 and no posizione and trend_up:
    entra long (BUY 95% USDT), logga su CSV, notifica Telegram
    salva entry_time per hold minimo

# SELL (segnale tecnico, non ML)
if sell_signal_tecnico and BTC > 0.0001 and posizione_aperta:
    chiudi long (SELL 95% BTC), logga P&L su CSV, notifica Telegram

altrimenti: HOLD

# Dashboard update (ogni ciclo, non-blocking)
salva snapshot su dashboard_data.json

# Retraining periodico (ogni RETRAIN_HOURS)
se ore_dall_ultimo_training >= RETRAIN_HOURS:
    scarica dati freschi, riallena modello, salva su disco

# Notifiche Telegram
BUY/SELL/STOP_LOSS/ERRORE → messaggio immediato
STATUS → 1 volta al giorno (STATUS_INTERVAL = 86400)
```

---

## Binance Demo Trading — come ottenerlo

1. Vai su **https://demo.binance.com/** (richiede account Binance reale)
2. Accedi con il tuo account Binance
3. Genera API Key dal pannello Demo Trading
4. Copia API Key e Secret nel file `.env`

**Nota**: le chiavi da `demo.binance.com` e `testnet.binance.vision` NON sono intercambiabili.
Il bot usa ccxt con `exchange.enable_demo_trading(True)`.

Il testnet fornisce automaticamente un wallet con BTC e USDT finti.
I dati di mercato usati dal bot vengono dalla Binance pubblica (reali).

---

## Dashboard di monitoraggio

```bash
# In un terminale separato:
python dashboard.py
# Apri http://localhost:5050 (o http://IP_VPS:5050)
```

Dashboard web (Flask + Tailwind CSS + Chart.js + Lightweight Charts) con:
- Cards: prezzo BTC (con sparkline), segnale, confidenza ML, saldo USDT/BTC, P&L
- Countdown bar al prossimo ciclo bot (aggiornamento ogni secondo)
- Flash + count-up animato sui numeri quando cambiano
- Grafico candlestick BTC/USDT (ultime 100 candele 1h, TradingView Lightweight Charts)
- Equity curve interattiva
- Radar delle feature ML
- Activity feed delle azioni recenti del bot
- Toast notification animati per nuovi trade
- Tabella ultimi 20 trade
- Dark theme, responsive, auto-refresh 30s

---

## Stato attuale del progetto

- [x] Fetch dati OHLCV paginato (10000 candele, ~416 giorni)
- [x] Feature engineering multi-timeframe (23 feature: 1h + 4h + 1d)
- [x] Ottimizzazione iperparametri Optuna (50 trial, cache 48h)
- [x] Binary classification (BUY vs NO-BUY) con scale_pos_weight
- [x] Target dinamico basato su ATR (riduce falsi segnali in alta volatilita')
- [x] SELL tramite regole tecniche confermate (RSI 75, MACD 2-bar, EMA 0.8%)
- [x] Walk-forward validation (~41 fold scorrevoli, fallback per fold degeneri)
- [x] Backtest con backtesting.py (stats + plot + P&L in $)
- [x] Stop loss automatico (2%) nel backtest e nel bot live
- [x] Hold minimo 5h per evitare uscite premature
- [x] Filtro trend EMA20 (blocca BUY in downtrend)
- [x] Loop live su Binance Testnet (solo long, no short su spot, ciclo 15 min)
- [x] Filtro confidenza su predict_proba (MIN_PROBA = 0.55)
- [x] Logging trade su CSV (trades_log.csv)
- [x] Feature importance analysis
- [x] Retraining periodico automatico (ogni 24h con dati freschi)
- [x] Persistenza modello su disco (joblib)
- [x] Persistenza stato posizione + entry_time (bot_state.json)
- [x] Notifiche Telegram (trade, stop loss, errori, retraining, status giornaliero)
- [x] Dashboard web (Flask + Chart.js + Tailwind, porta 5050)
- [x] Deploy su VPS (Hostinger Game Panel 4)

---

## Performance backtest (walk-forward, 342 giorni, B&H -3%)

| Metrica | Valore |
|---------|--------|
| Return | +27.7% |
| Sharpe | 1.25 |
| Sortino | 3.06 |
| Profit Factor | 1.83 |
| Win Rate | 48.8% |
| # Trade | 84 |
| Max Drawdown | -12.1% |
| Avg Trade | +0.32% |
| Avg Duration | 15h |
| SQN | 1.64 |

---

## Note importanti

- Il bot usa **Binance Testnet** per gli ordini: nessun soldo reale viene toccato.
- I dati OHLCV vengono scaricati dalla **Binance pubblica** (mercato reale).
- Il modello viene salvato su disco con `joblib` e ricaricato se ha meno di `RETRAIN_HOURS` ore.
- Lo stato della posizione (entry_price, entry_qty, entry_time) viene persistito in `bot_state.json`.
- `shuffle=False` nel train/test split e' fondamentale per time-series.
- La commission e' impostata a `0.001` (0.1%) nel backtest, uguale a Binance spot.
- Optuna best_params.json ha TTL 48h; cancellare per forzare ri-ottimizzazione.
- Fold degeneri usano il modello del fold precedente come fallback.
- `save_dashboard_data()` e' wrappato in try/except per non bloccare stop-loss in caso di errore I/O.

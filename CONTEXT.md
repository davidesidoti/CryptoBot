# cryptobot — Project Context for Claude Code

## Obiettivo del progetto

Bot di trading crypto basato su ML che opera su **Binance Testnet** (soldi finti, dati reali).
Il bot scarica candele OHLCV, genera feature tecniche, allena un classificatore XGBoost,
backtesta la strategia su dati storici, e infine gira in loop live piazzando ordini reali
sul testnet di Binance.

---

## Stack tecnico

| Componente         | Libreria / Servizio               |
|--------------------|-----------------------------------|
| Dati di mercato    | `ccxt` + Binance public API       |
| Feature engineering| `pandas-ta`                       |
| Modello ML         | `XGBClassifier` (xgboost)         |
| Backtest           | `backtesting.py`                  |
| Esecuzione ordini  | `ccxt` + Binance Testnet          |
| Linguaggio         | Python 3.10+                      |

### Install

```bash
pip install ccxt pandas numpy scikit-learn xgboost pandas-ta backtesting
```

---

## Architettura del bot

```
Binance public API (dati OHLCV storici, paginato per >1000 candele)
        ↓
build_features() → 20 feature (1h + 4h + 1d multi-timeframe) + target dinamico ATR
        ↓
walk_forward_train() → XGBoost binary (BUY vs NO-BUY), 17 fold scorrevoli
        ↓
backtest() → backtesting.py → stats + P&L in $ + plot interattivo
        ↓
run_bot() → loop live su Binance Testnet ogni SLEEP_SECONDS
```

---

## File esistente

**`cryptobot.py`** — unico file, tutto autocontenuto.

### Sezioni principali

1. **CONFIG** (in cima al file) — tutte le variabili configurabili
2. **`fetch_ohlcv()`** — scarica candele da Binance con paginazione (endpoint pubblico, no auth)
3. **`build_features()`** — indicatori 1h + multi-timeframe (4h, 1d) + target dinamico ATR
4. **`technical_sell_signal()`** — segnali SELL basati su regole tecniche (RSI, MACD, EMA)
5. **`train_model()`** — XGBoost binary (BUY vs NO-BUY), usato come fallback
6. **`walk_forward_train()`** — walk-forward con 17 fold scorrevoli, combina BUY ML + SELL tecnico
7. **`backtest()`** — strategia ML su backtesting.py con stop loss, trend filter, P&L in $
8. **`get_testnet_exchange()`** — istanza ccxt puntata al Binance Testnet
9. **`log_trade()`** — logging trade su CSV
10. **`run_bot()`** — loop live con BUY ML + SELL tecnico + stop loss + trend filter
11. **`__main__`** — esegue fetch → features → walk-forward → backtest;
   `run_bot()` è commentato, da decommentare per andare live

---

## Variabili di configurazione (CONFIG block)

```python
TESTNET_API_KEY = "YOUR_TESTNET_API_KEY"   # da testnet.binance.vision
TESTNET_SECRET  = "YOUR_TESTNET_SECRET"

SYMBOL          = "BTC/USDT"
TIMEFRAME       = "1h"        # timeframe delle candele
FETCH_LIMIT     = 5000        # quante candele storiche scaricare (paginato, ~208 giorni)
N_TRAIN         = 400         # (attualmente non usato nel loop, reserved)
FUTURE_BARS     = 3           # candele in avanti per calcolare il label
SIGNAL_THRESH   = 0.007       # soglia minima di movimento (0.7%), override da ATR dinamico
MIN_PROBA       = 0.60        # confidenza minima XGBoost per eseguire ordine BUY
TRADE_SIZE      = 0.95        # % del capitale usata per ogni ordine
STOP_LOSS       = 0.02        # 2% stop loss
INITIAL_CASH    = 500         # capitale iniziale per il backtest (in USD)
SLEEP_SECONDS   = 3600        # pausa tra cicli del bot (1h)
LOG_FILE        = "trades_log.csv"
RETRAIN_HOURS   = 24          # riaddestra il modello ogni N ore
MODEL_FILE      = "model.joblib"
STATE_FILE      = "bot_state.json"
WF_TRAIN_BARS   = 1500        # walk-forward: candele per finestra training
WF_TEST_BARS    = 200         # walk-forward: candele per finestra test
```

---

## Feature usate dal modello (20 totali)

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
    "rsi_slope",    # diff(RSI, 5) - momentum dell'RSI
    "hour",         # ora della candela (pattern intraday)
    # === Multi-timeframe (resample da 1h) ===
    "rsi_4h",       # RSI su candele 4h
    "macd_4h",      # MACD su candele 4h
    "ema_cross_4h", # EMA9-EMA21 su candele 4h
    "trend_4h",     # prezzo > EMA20 su 4h (0/1)
    "rsi_1d",       # RSI su candele daily
    "trend_1d",     # prezzo > EMA20 su daily (0/1)
    # === Regime / volatilita' ===
    "atr_ratio",    # ATR(7) / ATR(28) - >1 = vol in aumento
    "vol_regime"    # volume / SMA(volume, 20) - >1 = volume sopra media
]
```

---

## Label / target

**Binary classification**: BUY (1) vs NO-BUY (0).

- `1` (BUY) → prezzo sale > soglia dinamica nelle prossime 3 candele
- `0` (NO-BUY) → tutto il resto (HOLD + SELL)

La soglia e' **dinamica**, basata su ATR: `max(0.7%, ATR% * 0.5)`.
In mercati volatili la soglia si alza automaticamente, riducendo i falsi segnali.

Il segnale SELL e' gestito da **regole tecniche** (non ML):
- RSI > 70 e in discesa
- MACD cross ribassista
- Prezzo sotto EMA20 con momentum negativo

---

## Logica ordini nel bot live (solo LONG, Binance Spot)

```
# Stop loss (indipendente dal segnale ML)
if posizione_aperta and loss >= STOP_LOSS (2%):
    chiudi posizione, logga su CSV

# Filtro trend
if prezzo < EMA20: blocca tutti i BUY

# BUY (segnale ML binary)
if buy_proba >= MIN_PROBA (0.60) and USDT > $10 and no posizione and trend_up:
    entra long (BUY 95% USDT), logga su CSV

# SELL (segnale tecnico, non ML)
if sell_signal_tecnico and BTC > 0.0001 and posizione_aperta:
    chiudi long (SELL 95% BTC), logga P&L su CSV

altrimenti: HOLD

# Retraining periodico (ogni RETRAIN_HOURS)
se ore_dall_ultimo_training >= RETRAIN_HOURS:
    scarica dati freschi, riallena modello, salva su disco

# Notifiche Telegram
ogni BUY/SELL/STOP_LOSS/ERRORE → messaggio su Telegram
```

---

## Binance Testnet — come ottenerlo

1. Vai su **https://testnet.binance.vision/**
2. Login con GitHub
3. Clicca "Generate HMAC_SHA256 Key"
4. Copia API Key e Secret nel file `.env`

Il testnet fornisce automaticamente un wallet con BTC e USDT finti.
I dati di mercato usati dal bot vengono dalla Binance pubblica (reali).

---

## Stato attuale del progetto

- [x] Fetch dati OHLCV paginato (5000 candele, ~208 giorni)
- [x] Feature engineering multi-timeframe (20 feature: 1h + 4h + 1d)
- [x] Binary classification (BUY vs NO-BUY) con scale_pos_weight
- [x] Target dinamico basato su ATR (riduce falsi segnali in alta volatilita')
- [x] SELL tramite regole tecniche (RSI, MACD, EMA) anziche' ML
- [x] Walk-forward validation (17 fold scorrevoli, guard per fold degeneri)
- [x] Backtest con backtesting.py (stats + plot + P&L in $)
- [x] Stop loss automatico (2%) nel backtest e nel bot live
- [x] Filtro trend EMA20 (blocca BUY in downtrend)
- [x] Loop live su Binance Testnet (solo long, no short su spot)
- [x] Filtro confidenza su predict_proba (MIN_PROBA = 0.60)
- [x] Logging trade su CSV (trades_log.csv)
- [x] Feature importance analysis
- [x] Retraining periodico automatico (ogni 24h con dati freschi)
- [x] Persistenza modello su disco (joblib)
- [x] Persistenza stato posizione (bot_state.json)
- [x] Notifiche Telegram (trade, stop loss, errori, retraining)
- [ ] Dashboard di monitoraggio (P&L, trade history, equity curve)
- [ ] Deploy su server remoto (VPS)

---

## Possibili miglioramenti prioritari

### 1. Dashboard di monitoraggio
Web dashboard per visualizzare P&L, trade history, equity curve in tempo reale.

### 2. Deploy su VPS
Rendere il bot eseguibile su un server remoto con supervisord o systemd.

---

## Note importanti

- Il bot usa **Binance Testnet** per gli ordini: nessun soldo reale viene toccato.
- I dati OHLCV vengono scaricati dalla **Binance pubblica** (mercato reale).
- Il modello viene salvato su disco con `joblib` dopo il training e ricaricato automaticamente se ha meno di `RETRAIN_HOURS` ore.
- Lo stato della posizione (entry_price, entry_qty) viene persistito in `bot_state.json` per sopravvivere ai restart.
- `shuffle=False` nel train/test split e' fondamentale per time-series.
- La commission e' impostata a `0.001` (0.1%) nel backtest, uguale a Binance spot.

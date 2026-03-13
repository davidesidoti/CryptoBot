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
Binance public API (dati OHLCV storici)
        ↓
build_features() → RSI, MACD, BB, EMA cross, vol change, price change
        ↓
train_model() → XGBoost multiclasse (SELL=0 / HOLD=1 / BUY=2)
        ↓
backtest() → backtesting.py → stats + plot interattivo
        ↓
run_bot() → loop live su Binance Testnet ogni SLEEP_SECONDS
```

---

## File esistente

**`cryptobot.py`** — unico file, tutto autocontenuto.

### Sezioni principali

1. **CONFIG** (in cima al file) — tutte le variabili configurabili
2. **`fetch_ohlcv()`** — scarica candele da Binance (endpoint pubblico, no auth)
3. **`build_features()`** — aggiunge indicatori tecnici + label supervisionato
4. **`train_model()`** — allena XGBoost, stampa classification report
5. **`backtest()`** — strategia ML su backtesting.py, mostra stats e plot
6. **`get_testnet_exchange()`** — istanza ccxt puntata al Binance Testnet
7. **`run_bot()`** — loop live con gestione ordini e filtro confidenza
8. **`__main__`** — esegue fetch → features → train → backtest in sequenza;
   `run_bot()` è commentato, da decommentare per andare live

---

## Variabili di configurazione (CONFIG block)

```python
TESTNET_API_KEY = "YOUR_TESTNET_API_KEY"   # da testnet.binance.vision
TESTNET_SECRET  = "YOUR_TESTNET_SECRET"

SYMBOL          = "BTC/USDT"
TIMEFRAME       = "1h"        # timeframe delle candele
FETCH_LIMIT     = 500         # quante candele storiche scaricare
N_TRAIN         = 400         # (attualmente non usato nel loop, reserved)
FUTURE_BARS     = 3           # candele in avanti per calcolare il label
SIGNAL_THRESH   = 0.005       # soglia di movimento (0.5%) per BUY/SELL label
MIN_PROBA       = 0.50        # confidenza minima XGBoost per eseguire ordine
TRADE_SIZE      = 0.95        # % del capitale usata per ogni ordine
SLEEP_SECONDS   = 3600        # pausa tra cicli del bot (1h)
```

---

## Feature usate dal modello

```python
FEATURES = [
    "rsi",          # RSI 14
    "macd",         # MACD line (12,26,9)
    "macd_signal",  # MACD signal line
    "bb_width",     # Larghezza Bollinger Bands normalizzata
    "vol_change",   # Variazione % volume candela precedente
    "price_change", # Variazione % prezzo candela precedente
    "ema_cross"     # EMA9 - EMA21 (differenza)
]
```

---

## Label / target

- `+1` (BUY)  → prezzo sale > 0.5% nelle prossime 3 candele
- `-1` (SELL) → prezzo scende > 0.5% nelle prossime 3 candele
- `0`  (HOLD) → movimento sotto soglia

XGBoost riceve classi `0, 1, 2` (shift di +1), il bot riconverte a `-1, 0, 1`.

---

## Logica ordini nel bot live

```
if signal == BUY and confidenza >= MIN_PROBA and USDT disponibile:
    piazza ordine market BUY per il 95% dell'USDT disponibile

if signal == SELL and confidenza >= MIN_PROBA and BTC disponibile:
    piazza ordine market SELL per il 95% del BTC disponibile

altrimenti: skip (HOLD o confidenza troppo bassa)
```

---

## Binance Testnet — come ottenerlo

1. Vai su **https://testnet.binance.vision/**
2. Login con GitHub
3. Clicca "Generate HMAC_SHA256 Key"
4. Copia API Key e Secret nel CONFIG block del file

Il testnet fornisce automaticamente un wallet con BTC e USDT finti.
I dati di mercato usati dal bot vengono dalla Binance pubblica (reali).

---

## Stato attuale del progetto

- [x] Fetch dati OHLCV funzionante
- [x] Feature engineering completo
- [x] Training XGBoost con classification report
- [x] Backtest con backtesting.py (stats + plot)
- [x] Loop live su Binance Testnet con gestione ordini
- [x] Filtro confidenza su predict_proba
- [ ] Walk-forward validation (non ancora implementata)
- [ ] Stop loss automatico per trade
- [ ] Retraining periodico del modello con dati freschi
- [ ] Logging su file / database
- [ ] Notifiche (Telegram, email) su ogni ordine
- [ ] Dashboard di monitoraggio (P&L, trade history, equity curve)
- [ ] Supporto multi-symbol / multi-timeframe
- [ ] Deploy su server remoto (VPS)

---

## Possibili miglioramenti prioritari

### 1. Walk-forward validation
Invece di un semplice train/test split, splittare i dati in finestre
temporali scorrevoli per evitare overfitting su serie storiche.

### 2. Stop loss
Aggiungere un controllo nel loop che chiude la posizione se il prezzo
scende di X% rispetto al prezzo di entrata, indipendentemente dal segnale ML.

### 3. Retraining automatico
Ogni N ore, riscaricare dati freschi e rifare il fit del modello
per evitare che diventi stale nel tempo.

### 4. Logging strutturato
Salvare ogni trade su un file CSV o SQLite con:
timestamp, symbol, side, qty, price, signal, confidence, pnl.

### 5. Notifiche Telegram
Usare la Telegram Bot API per ricevere un messaggio ogni volta
che il bot piazza un ordine.

---

## Note importanti

- Il bot usa **Binance Testnet** per gli ordini: nessun soldo reale viene toccato.
- I dati OHLCV vengono scaricati dalla **Binance pubblica** (mercato reale).
- Il modello viene allenato ad ogni avvio da zero — non c'e' persistenza del modello.
- Per persistere il modello tra un avvio e l'altro: usare `joblib.dump` / `joblib.load`.
- `shuffle=False` nel train/test split e' fondamentale per time-series.
- La commission e' impostata a `0.001` (0.1%) nel backtest, uguale a Binance spot.

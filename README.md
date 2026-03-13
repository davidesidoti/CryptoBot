# CryptoBot

Bot di trading crypto basato su Machine Learning che opera su **Binance Testnet** (soldi finti, dati di mercato reali).

Scarica candele OHLCV, genera feature tecniche multi-timeframe, allena un classificatore XGBoost con walk-forward validation, backtesta la strategia su dati storici, e gira in loop live piazzando ordini sul testnet di Binance.

## Come funziona

```
Binance API (dati OHLCV reali, 5000 candele ~ 208 giorni)
        |
build_features() - 20 feature (1h + 4h + 1d multi-timeframe) + target dinamico ATR
        |
walk_forward_train() - XGBoost binary (BUY vs NO-BUY), 17 fold scorrevoli
        |
backtest() - backtesting.py - stats + P&L + plot interattivo
        |
run_bot() - loop live su Binance Testnet ogni ora
```

**Segnali BUY** generati dal modello ML (XGBoost) con filtro di confidenza.
**Segnali SELL** generati da regole tecniche (RSI, MACD, EMA) - non dal modello ML.
**Stop loss** automatico al 2%.

## Quick Start

### 1. Clona e installa

```bash
git clone https://github.com/YOUR_USERNAME/CryptoBot.git
cd CryptoBot
python -m venv .venv
source .venv/Scripts/activate   # Windows bash
pip install -r requirements.txt
```

### 2. Configura le credenziali

```bash
cp .env.example .env
```

Inserisci le tue API key di Binance Testnet nel file `.env`:
- Vai su https://testnet.binance.vision/
- Accedi con GitHub
- Clicca "Generate HMAC_SHA256 Key"
- Copia API Key e Secret nel `.env`

Per le notifiche Telegram (opzionale):
- Crea un bot su [@BotFather](https://t.me/BotFather) e copia il token
- Avvia il bot, poi visita `https://api.telegram.org/bot<TOKEN>/getUpdates` per trovare il `chat_id`

### 3. Lancia il backtest

```bash
python cryptobot.py
```

Il bot esegue automaticamente: download dati - feature engineering - walk-forward training - backtest.

### 4. Attiva il live trading (testnet)

Apri `cryptobot.py` e decommenta l'ultima riga nel blocco `__main__`:

```python
run_bot(model)
```

Il bot girera' in loop, piazzando ordini sul testnet ogni ora.

## Configurazione

Tutte le variabili sono nel blocco CONFIG in cima a `cryptobot.py`:

| Variabile | Default | Descrizione |
|-----------|---------|-------------|
| `SYMBOL` | `BTC/USDT` | Coppia di trading |
| `TIMEFRAME` | `1h` | Timeframe delle candele |
| `FETCH_LIMIT` | `5000` | Candele storiche da scaricare |
| `MIN_PROBA` | `0.60` | Confidenza minima XGBoost per BUY |
| `TRADE_SIZE` | `0.95` | % del capitale per ogni ordine |
| `STOP_LOSS` | `0.02` | Stop loss (2%) |
| `INITIAL_CASH` | `500` | Capitale iniziale per il backtest (USD) |
| `RETRAIN_HOURS` | `24` | Riaddestra il modello ogni N ore |
| `SLEEP_SECONDS` | `3600` | Pausa tra i cicli del bot (1h) |

## Feature del modello (20 totali)

Il modello usa feature su tre timeframe:

- **1h (base):** RSI, MACD, MACD signal, Bollinger Width, volume change, price change, EMA cross, ATR, OBV change, Stochastic K, RSI slope, hour
- **4h (resampled):** RSI, MACD, EMA cross, trend
- **1d (resampled):** RSI, trend
- **Regime:** ATR ratio (7/28), volume regime

## Architettura ML

- **Classificazione binaria:** BUY (1) vs NO-BUY (0)
- **Target dinamico:** soglia basata su ATR - `max(0.7%, ATR% * 0.5)`
- **Walk-forward validation:** 17 fold scorrevoli (train=1500, test=200 candele)
- **Degenerate fold guard:** se `best_iteration < 10`, tutte le predizioni diventano NO-BUY
- **Filtro trend:** BUY bloccati se prezzo < EMA20

## Funzionalita'

- [x] Download dati OHLCV paginato (5000 candele, ~208 giorni)
- [x] Feature engineering multi-timeframe (20 feature: 1h + 4h + 1d)
- [x] Walk-forward validation con 17 fold scorrevoli
- [x] Backtest con backtesting.py (stats + plot + P&L)
- [x] SELL tramite regole tecniche (RSI, MACD, EMA)
- [x] Stop loss automatico (2%)
- [x] Filtro trend EMA20
- [x] Filtro confidenza (MIN_PROBA = 0.60)
- [x] Loop live su Binance Testnet
- [x] Retraining automatico ogni 24h
- [x] Persistenza modello su disco (joblib)
- [x] Persistenza stato posizione (bot_state.json)
- [x] Notifiche Telegram (trade, stop loss, errori, retraining)
- [x] Logging trade su CSV

## Stack tecnico

| Componente | Libreria |
|---|---|
| Dati di mercato | `ccxt` + Binance API |
| Feature engineering | `pandas-ta` |
| Modello ML | `XGBClassifier` (xgboost) |
| Backtest | `backtesting.py` |
| Esecuzione ordini | `ccxt` + Binance Testnet |
| Notifiche | Telegram Bot API |

## Struttura file

```
CryptoBot/
  cryptobot.py      # bot completo (unico file)
  .env.example       # template per le credenziali
  requirements.txt   # dipendenze Python
  .gitignore
  CONTEXT.md         # documentazione tecnica dettagliata
  README.md
```

File generati a runtime (non committati):
- `model.joblib` - modello XGBoost salvato
- `bot_state.json` - stato posizione corrente
- `trades_log.csv` - log di tutti i trade
- `MLStrategy.html` - plot interattivo del backtest

## Disclaimer

Questo bot e' un progetto educativo. Anche se opera su Binance Testnet (soldi finti), le performance passate non garantiscono risultati futuri. Non usare con soldi reali senza comprendere i rischi del trading algoritmico.

## License

MIT

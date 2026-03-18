# CryptoBot

Bot di trading crypto basato su Machine Learning che opera su **Binance Testnet** (soldi finti, dati di mercato reali).

Scarica candele OHLCV, genera 23 feature tecniche multi-timeframe, ottimizza iperparametri con Optuna, allena un classificatore XGBoost con walk-forward validation (~41 fold), backtesta la strategia, e gira in loop live piazzando ordini sul testnet di Binance. Include dashboard web per monitoraggio real-time.

## Come funziona

```
Binance API (dati OHLCV reali, 10000 candele ~ 416 giorni)
        |
build_features() - 23 feature (1h + 4h + 1d multi-timeframe) + target dinamico ATR
        |
optimize_hyperparams() - Optuna bayesian search (50 trial, cache 48h)
        |
walk_forward_train() - XGBoost binary (BUY vs NO-BUY), ~41 fold scorrevoli
        |
backtest() - backtesting.py - stats + P&L + plot interattivo
        |
run_bot() - loop live su Binance Testnet ogni 15 min
```

**Segnali BUY** generati dal modello ML (XGBoost) con filtro di confidenza (>55%).
**Segnali SELL** generati da regole tecniche (RSI > 75, MACD 2-bar, EMA20 break) - non dal modello ML.
**Stop loss** automatico al 2%, sempre attivo.
**Hold minimo** 5h prima di SELL tecnico, per evitare uscite premature.

## Performance (backtest walk-forward, 342 giorni)

| Metrica | Valore |
|---------|--------|
| Return | **+27.7%** (vs B&H -3%) |
| Sharpe | 1.25 |
| Sortino | 3.06 |
| Profit Factor | 1.83 |
| Win Rate | 48.8% |
| Trade | 84 |
| Max Drawdown | -12.1% |

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

Inserisci le tue API key di Binance Demo Trading nel file `.env`:
- Vai su https://demo.binance.com/ (richiede account Binance reale)
- Accedi con il tuo account Binance
- Genera API Key dal pannello Demo Trading
- Copia API Key e Secret nel `.env`

Per le notifiche Telegram (opzionale):
- Crea un bot su [@BotFather](https://t.me/BotFather) e copia il token
- Avvia il bot, poi visita `https://api.telegram.org/bot<TOKEN>/getUpdates` per trovare il `chat_id`

### 3. Lancia il backtest

```bash
python cryptobot.py
```

Il bot esegue automaticamente: download dati → ottimizzazione Optuna → walk-forward training → backtest.

### 4. Attiva il live trading (testnet)

Apri `cryptobot.py` e decommenta l'ultima riga nel blocco `__main__`:

```python
run_bot(model)
```

### 5. Avvia la dashboard (opzionale)

```bash
# In un terminale separato:
python dashboard.py
# Apri http://localhost:5050
```

## Configurazione

Tutte le variabili sono nel blocco CONFIG in cima a `cryptobot.py`:

| Variabile | Default | Descrizione |
|-----------|---------|-------------|
| `SYMBOL` | `BTC/USDT` | Coppia di trading |
| `TIMEFRAME` | `1h` | Timeframe delle candele |
| `FETCH_LIMIT` | `10000` | Candele storiche da scaricare (~416 giorni) |
| `FUTURE_BARS` | `5` | Candele in avanti per il label |
| `MIN_PROBA` | `0.55` | Confidenza minima XGBoost per BUY |
| `TRADE_SIZE` | `0.95` | % del capitale per ogni ordine |
| `STOP_LOSS` | `0.02` | Stop loss (2%) |
| `MIN_HOLD_BARS` | `5` | Hold minimo 5h prima di SELL tecnico |
| `INITIAL_CASH` | `500` | Capitale iniziale per il backtest (USD) |
| `RETRAIN_HOURS` | `24` | Riaddestra il modello ogni N ore |
| `SLEEP_SECONDS` | `900` | Pausa tra i cicli del bot (15 min) |
| `MAX_RETRIES` | `3` | Tentativi per errori di rete transitori |
| `RETRY_BACKOFF` | `[30, 60, 120]` | Secondi di attesa tra retry |
| `OPTUNA_TRIALS` | `50` | Trial per ottimizzazione bayesiana |

## Feature del modello (23 totali)

Il modello usa feature su tre timeframe:

- **1h (base):** RSI, MACD, MACD signal, Bollinger Width, volume change, price change, EMA cross, ATR, OBV change, Stochastic K, RSI slope, hour, ADX, Williams %R, VWAP distance
- **4h (resampled):** RSI, MACD, EMA cross, trend
- **1d (resampled):** RSI, ADX
- **Regime:** ATR ratio (7/28), volume regime

## Architettura ML

- **Classificazione binaria:** BUY (1) vs NO-BUY (0)
- **Target dinamico:** soglia basata su ATR — `max(0.7%, ATR% * 0.5)`
- **Ottimizzazione Optuna:** 50 trial bayesiani, cache su disco per 48h
- **Walk-forward validation:** ~41 fold scorrevoli (train=1500, test=200 candele)
- **Degenerate fold guard:** usa modello del fold precedente come fallback
- **Filtro trend:** BUY bloccati se prezzo < EMA20
- **Hold minimo:** SELL tecnico soppresso per 5h dopo BUY (stop loss sempre attivo)

## Funzionalita'

- [x] Download dati OHLCV paginato (10000 candele, ~416 giorni)
- [x] Feature engineering multi-timeframe (23 feature: 1h + 4h + 1d)
- [x] Ottimizzazione iperparametri Optuna (50 trial, cache 48h)
- [x] Walk-forward validation con ~41 fold scorrevoli
- [x] Backtest con backtesting.py (stats + plot + P&L)
- [x] SELL tramite regole tecniche confermate (RSI 75, MACD 2-bar, EMA 0.8%)
- [x] Stop loss automatico (2%)
- [x] Hold minimo 5h
- [x] Filtro trend EMA20
- [x] Filtro confidenza (MIN_PROBA = 0.55)
- [x] Loop live su Binance Testnet (ciclo 15 min)
- [x] Retraining automatico ogni 24h
- [x] Persistenza modello su disco (joblib)
- [x] Persistenza stato posizione + entry_time (bot_state.json)
- [x] Retry con backoff per errori di rete (3 tentativi, 30/60/120s)
- [x] Notifiche Telegram (trade, stop loss, errori, retraining, status giornaliero)
- [x] Logging trade su CSV
- [x] Dashboard web (Flask + Chart.js + Tailwind, porta 5050)
- [x] Deploy su VPS

## Stack tecnico

| Componente | Libreria |
|---|---|
| Dati di mercato | `ccxt` + Binance API |
| Feature engineering | `pandas-ta` |
| Modello ML | `XGBClassifier` (xgboost) |
| Ottimizzazione HP | `optuna` |
| Backtest | `backtesting.py` |
| Esecuzione ordini | `ccxt` + Binance Testnet |
| Dashboard | `flask` + Chart.js + Tailwind CSS |
| Notifiche | Telegram Bot API |

## Struttura file

```
CryptoBot/
  cryptobot.py          # bot completo (unico file)
  dashboard.py          # dashboard web Flask (porta 5050)
  templates/
    dashboard.html      # template dashboard (dark theme, Chart.js)
  .env.example          # template per le credenziali
  requirements.txt      # dipendenze Python
  .gitignore
  CONTEXT.md            # documentazione tecnica dettagliata
  CLAUDE.md             # istruzioni per Claude Code
  README.md
```

File generati a runtime (non committati):
- `model.joblib` — modello XGBoost salvato
- `best_params.json` — iperparametri Optuna (cache 48h)
- `bot_state.json` — stato posizione corrente + entry_time
- `dashboard_data.json` — snapshot ciclo per la dashboard
- `trades_log.csv` — log di tutti i trade
- `MLStrategy.html` — plot interattivo del backtest

## Disclaimer

Questo bot e' un progetto educativo. Anche se opera su Binance Testnet (soldi finti), le performance passate non garantiscono risultati futuri. Non usare con soldi reali senza comprendere i rischi del trading algoritmico.

## License

MIT

# CryptoBot

Bot di trading crypto basato su Machine Learning che opera su **Binance Demo Trading** (soldi finti, dati di mercato reali).

Dual-model **LONG + SHORT**: scarica candele OHLCV, genera 25 feature tecniche multi-timeframe, ottimizza iperparametri con Optuna (separatamente per BUY e SHORT), allena due classificatori XGBoost con walk-forward validation (~41 fold), backtesta la strategia, e gira in loop live piazzando ordini LONG su Spot Demo e SHORT su Futures Demo. Include dashboard web con signal log dettagliato.

## Come funziona

```
Binance API (dati OHLCV reali, 10000 candele ~ 416 giorni)
        |
build_features() - 25 feature (1h + 4h + 1d multi-timeframe) + target dinamico ATR
        |
optimize_hyperparams() - 2x Optuna bayesian search (50 trial ciascuno, cache 48h)
        |
walk_forward_train() - dual XGBoost (BUY + SHORT), ~41 fold scorrevoli
        |
backtest() - backtesting.py LONG+SHORT - stats + P&L + plot interattivo
        |
run_bot() - loop live: LONG su Spot Demo + SHORT su Futures Demo, ogni 15 min
```

**Segnali BUY** generati dal modello ML (XGBoost) con filtro di confidenza (>55%) + trend filter (prezzo > EMA20).
**Segnali SHORT** generati dal secondo modello ML con filtro di confidenza (>60%) + trend filter (prezzo < EMA20).
**Chiusura LONG** tramite regole tecniche (RSI > 75, MACD 2-bar, EMA20 break) - non dal modello ML.
**Chiusura SHORT** tramite regole tecniche cover (RSI < 30, MACD bullish, prezzo > EMA20).
**Stop loss** automatico al 2% per LONG e SHORT, sempre attivo.
**Hold minimo** 5h per LONG, 3h per SHORT, per evitare uscite premature.

## Performance (backtest walk-forward LONG+SHORT, 342 giorni)

| Metrica | LONG+SHORT | B&H |
|---------|-----------|-----|
| Return | **+67.6%** | -8.9% |
| Sharpe | **2.04** | — |
| Sortino | **6.70** | — |
| Calmar | **7.05** | — |
| Profit Factor | 1.87 | — |
| Win Rate | 48.5% | — |
| Trade | 194 | — |
| Max Drawdown | -10.1% | — |
| Avg Trade | +0.35% | — |
| Exposure | 38.3% | 100% |
| SQN | 2.48 | — |
| P&L netto | **+$337.91** (su $500) | — |

Alpha: **+76 punti percentuali** rispetto a Buy & Hold.

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
- **Assicurati che "Enable Futures" sia spuntato** (per abilitare SHORT)
- Copia API Key e Secret nel `.env`

Per le notifiche Telegram (opzionale):
- Crea un bot su [@BotFather](https://t.me/BotFather) e copia il token
- Avvia il bot, poi visita `https://api.telegram.org/bot<TOKEN>/getUpdates` per trovare il `chat_id`

### 3. Lancia il backtest

```bash
python cryptobot.py
```

Il bot esegue: download dati -> ottimizzazione Optuna (BUY + SHORT) -> walk-forward training -> backtest LONG+SHORT.

```bash
# Per rieseguire solo il backtest senza retraining:
python cryptobot.py --backtest
```

### 4. Attiva il live trading (demo)

Apri `cryptobot.py` e decommenta l'ultima riga nel blocco `__main__`:

```python
run_bot(model_buy, model_short)
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
| `SHORT_MIN_PROBA` | `0.60` | Confidenza minima per SHORT (piu' alta) |
| `TRADE_SIZE` | `0.95` | % del capitale per LONG |
| `SHORT_TRADE_SIZE` | `0.70` | % del capitale per SHORT (piu' conservativo) |
| `STOP_LOSS` | `0.02` | Stop loss LONG (2%) |
| `SHORT_STOP_LOSS` | `0.02` | Stop loss SHORT (2%) |
| `MIN_HOLD_BARS` | `5` | Hold minimo 5h prima di CLOSE LONG |
| `SHORT_MIN_HOLD` | `3` | Hold minimo 3h prima di CLOSE SHORT |
| `INITIAL_CASH` | `500` | Capitale iniziale per il backtest (USD) |
| `RETRAIN_HOURS` | `24` | Riaddestra i modelli ogni N ore |
| `SLEEP_SECONDS` | `900` | Pausa tra i cicli del bot (15 min) |
| `OPTUNA_TRIALS` | `50` | Trial per ottimizzazione bayesiana (per modello) |
| `ENABLE_SHORT` | `True` | Abilita/disabilita modello e trading SHORT |

## Feature del modello (25 totali)

Entrambi i modelli usano le stesse feature su tre timeframe:

- **1h (base):** RSI, MACD, MACD signal, MACD hist, Bollinger Width, volume change, price change, EMA cross, ATR, OBV change, Stochastic K, RSI slope, hour, ADX, Williams %R, VWAP distance
- **4h (resampled):** RSI, MACD, EMA cross, trend
- **1d (resampled):** RSI, ADX
- **Regime:** ATR ratio (7/28), volume regime
- **Trend:** trend_down (prezzo < EMA20)

## Architettura ML

- **Dual classification:** BUY model (BUY vs NO-BUY) + SHORT model (SHORT vs NO-SHORT)
- **Target dinamico:** soglia basata su ATR — `max(0.7%, ATR% * 0.5)`
- **Ottimizzazione Optuna:** 50 trial bayesiani per modello, cache su disco 48h
- **Walk-forward validation:** ~41 fold scorrevoli (train=1500, test=200 candele)
- **Degenerate fold guard:** usa modello del fold precedente come fallback
- **Filtro trend:** BUY bloccati in downtrend, SHORT bloccati in uptrend
- **Conflict resolution:** se BUY e SHORT sparano insieme -> HOLD
- **Hold minimo:** CLOSE LONG soppresso per 5h, CLOSE SHORT per 3h (stop loss sempre attivo)

## Dashboard

La dashboard web include:
- **Decisione Corrente:** azione presa + motivo dettagliato + trend EMA20 + posizione
- **Modelli ML:** barre di confidenza BUY e SHORT con soglie visualizzate
- **Signal Log:** cronologia scrollabile ultimi 50 cicli con motivo di ogni decisione
- Grafico candlestick BTC/USDT (Lightweight Charts)
- Equity curve, radar feature, activity feed, tabella trade
- Toast notification per nuovi trade, countdown al prossimo ciclo

## Funzionalita'

- [x] Download dati OHLCV paginato (10000 candele, exchange cached + fallback)
- [x] Feature engineering multi-timeframe (25 feature: 1h + 4h + 1d)
- [x] Dual model BUY + SHORT con Optuna separata
- [x] Walk-forward validation ~41 fold con fallback degeneri
- [x] Backtest LONG+SHORT con backtesting.py
- [x] Stop loss 2% per LONG e SHORT
- [x] Hold minimo: 5h LONG, 3h SHORT
- [x] Filtro trend EMA20 + conflict resolution
- [x] Loop live: LONG su Spot Demo + SHORT su Futures Demo (leva 1x)
- [x] try/except su ogni ordine, no double-trade per ciclo
- [x] Retraining automatico ogni 24h
- [x] Persistenza modello (dual joblib) e stato (position_type)
- [x] Retry con backoff per errori di rete (3 tentativi)
- [x] Notifiche Telegram (trade, stop loss, errori, status giornaliero)
- [x] Dashboard web con signal log (action + reason per ogni ciclo)
- [x] Flag `--backtest` per skip training
- [x] Deploy su VPS

## Stack tecnico

| Componente | Libreria |
|---|---|
| Dati di mercato | `ccxt` + Binance API |
| Feature engineering | `pandas-ta` |
| Modello ML | `XGBClassifier` (xgboost) x2 |
| Ottimizzazione HP | `optuna` |
| Backtest | `backtesting.py` |
| Esecuzione ordini | `ccxt` + Binance Demo (Spot + Futures) |
| Dashboard | `flask` + Chart.js + Tailwind CSS |
| Notifiche | Telegram Bot API |

## Struttura file

```
CryptoBot/
  cryptobot.py          # bot completo (dual model LONG+SHORT)
  dashboard.py          # dashboard web Flask (porta 5050)
  templates/
    dashboard.html      # template dashboard (dark theme, signal log)
  .env.example          # template per le credenziali
  requirements.txt      # dipendenze Python
  .gitignore
  CONTEXT.md            # documentazione tecnica dettagliata
  CLAUDE.md             # istruzioni per Claude Code
  README.md
```

File generati a runtime (non committati):
- `model_buy.joblib` — modello XGBoost BUY
- `model_short.joblib` — modello XGBoost SHORT
- `best_params_buy.json` — iperparametri Optuna BUY (cache 48h)
- `best_params_short.json` — iperparametri Optuna SHORT (cache 48h)
- `bot_state.json` — stato posizione + position_type
- `dashboard_data.json` — snapshot ciclo + signal_log per dashboard
- `trades_log.csv` — log di tutti i trade
- `price_history.json` — ultime 100 candele per chart
- `MLStrategy.html` — plot interattivo del backtest

## Disclaimer

Questo bot e' un progetto educativo. Anche se opera su Binance Demo Trading (soldi finti), le performance passate non garantiscono risultati futuri. Non usare con soldi reali senza comprendere i rischi del trading algoritmico.

## License

MIT

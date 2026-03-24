# CryptoBot Scalping

Bot di scalping crypto basato su Machine Learning che opera su **Binance Futures Demo** (soldi finti, dati di mercato reali).

Dual-model **LONG + SHORT** su **candele 5m**: scarica OHLCV, genera 28 feature tecniche ottimizzate per scalping (5m + 15m + 1h multi-timeframe), ottimizza iperparametri con Optuna, allena due classificatori XGBoost con walk-forward validation (~20 fold), backtesta la strategia, e gira in loop live piazzando ordini LONG e SHORT su **Futures Demo** (fee 0.02% maker — critico per scalping). Include dashboard web con signal log dettagliato.

> **Branch `scalping`** — versione scalping del bot. Il branch `main` contiene la versione swing trading su candele 1h.

## Come funziona

```
Binance API (dati OHLCV reali, 10000 candele 5m ~ 34 giorni)
        |
build_features() - 28 feature scalping (5m + 15m + 1h multi-timeframe)
        |
optimize_hyperparams() - 2x Optuna bayesian search (60 trial ciascuno, cache 48h)
        |
walk_forward_train() - dual XGBoost (BUY + SHORT), ~20 fold scorrevoli
        |
backtest() - backtesting.py LONG+SHORT - TP 0.6%, SL 0.4%, trail ATR×1.5
        |
run_bot() - loop live: LONG + SHORT su Futures Demo, ciclo 60s (segnali ogni 5m)
```

**Segnali BUY** generati dal modello ML (XGBoost) con filtro di confidenza (>58%) + trend 1h UP.
**Segnali SHORT** generati dal secondo modello ML con filtro di confidenza (>60%) + trend 1h DOWN + ADX 1h > 15.
**Chiusura**: Take profit 0.6%, trailing stop ATR×1.5 (attiva dopo +0.3%), stop loss 0.4%, oppure segnali tecnici (RSI/MACD/EMA).
**Hold minimo**: 10 minuti (2 barre da 5m) per evitare uscite premature.
**Futures-only**: entrambe le direzioni passano per Binance Futures Demo (fee 0.02% vs 0.1% spot).

## Differenze dal branch main

| Aspetto | main (swing) | scalping |
|---------|-------------|----------|
| Timeframe | 1h | **5m** |
| Feature | 30 (1h+4h+1d+15m) | **28 (5m+15m+1h)** |
| Dati storici | ~416 giorni | **~34 giorni** |
| Ciclo bot | 15 min | **60s** (segnali ML ogni 5m) |
| Take Profit | nessuno | **0.6%** |
| Stop Loss | 2% | **0.4%** |
| Trailing | ATR×3, attiva +1% | **ATR×1.5, attiva +0.3%** |
| Hold minimo | 5h LONG, 3h SHORT | **10 min entrambi** |
| Commissione | 0.1% (Spot) | **0.04% (Futures)** |
| Exchange | Spot + Futures | **Futures-only** |
| Walk-forward | ~41 fold | **~20 fold** |
| Retrain | 24h | **12h** |

## Performance (backtest walk-forward LONG+SHORT)

> Esegui `python cryptobot.py` per ottenere i risultati aggiornati.

## Quick Start

### 1. Clona e installa

```bash
git clone https://github.com/YOUR_USERNAME/CryptoBot.git
cd CryptoBot
git checkout scalping
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
- Genera API Key dal pannello Demo Trading
- **Assicurati che "Enable Futures" sia spuntato** (obbligatorio per questa versione)
- Copia API Key e Secret nel `.env`

### 3. Lancia il backtest

```bash
python cryptobot.py
```

Il bot esegue: download dati 5m -> ottimizzazione Optuna (BUY + SHORT) -> walk-forward training -> backtest scalping.

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
python dashboard.py
# Apri http://localhost:5050
```

## Configurazione

| Variabile | Default | Descrizione |
|-----------|---------|-------------|
| `SYMBOL` | `BTC/USDT` | Coppia di trading |
| `TIMEFRAME` | `5m` | Timeframe candele (scalping) |
| `FETCH_LIMIT` | `10000` | Candele storiche (~34 giorni a 5m) |
| `FUTURE_BARS` | `3` | Candele lookahead per il label (15 min) |
| `MIN_PROBA` | `0.58` | Confidenza minima BUY |
| `SHORT_MIN_PROBA` | `0.60` | Confidenza minima SHORT |
| `TRADE_SIZE` | `0.80` | % del capitale per trade |
| `TAKE_PROFIT` | `0.006` | Take profit 0.6% |
| `STOP_LOSS` | `0.004` | Stop loss 0.4% |
| `SHORT_STOP_LOSS` | `0.004` | Stop loss SHORT 0.4% |
| `TRAIL_ATR_MULT` | `1.5` | Trailing stop = ATR × 1.5 |
| `TRAIL_ACTIVATE` | `0.003` | Trailing attiva dopo +0.3% |
| `MIN_HOLD_BARS` | `2` | Hold minimo 2 barre = 10 min |
| `INITIAL_CASH` | `500` | Capitale iniziale backtest (USD) |
| `RETRAIN_HOURS` | `12` | Riaddestra ogni 12 ore |
| `SLEEP_SECONDS` | `60` | Ciclo bot (60s, segnali ML ogni 5m) |
| `OPTUNA_TRIALS` | `60` | Trial per ottimizzazione bayesiana |
| `USE_FUTURES_FOR_BOTH` | `True` | LONG e SHORT su Futures (fee 0.02%) |

## Feature del modello (28 totali)

Entrambi i modelli usano le stesse feature su tre timeframe:

- **5m base - oscillatori/momentum (11):** RSI fast(7), RSI slope, MACD fast(8,21,5) + signal + histogram, Stochastic K/D(5,3,3), Bollinger width(10), Bollinger %B, Williams %R(7), CCI(14)
- **5m dinamica prezzo/volume (7):** price change (1-bar, 3-bar), vol change, vol spike, VWAP distance, ATR(10), ATR ratio(5/20)
- **5m microstruttura (3):** spread proxy (range/close), body ratio, EMA cross fast(5-20)
- **15m contesto (3):** RSI, MACD, trend (> EMA20)
- **1h contesto (3):** RSI, trend (> EMA20), ADX
- **Tempo (1):** minute_of_day (0-1439)

## Perche' Futures-only

Lo scalping genera profitti piccoli per trade (0.3-0.6%). Le commissioni sono il fattore critico:

| Exchange | Fee per lato | Round-trip | Impatto su 0.5% profit |
|----------|-------------|------------|------------------------|
| Spot | 0.10% | 0.20% | 40% del profitto |
| **Futures** | **0.02%** | **0.04%** | **8% del profitto** |

Usare Futures riduce l'impatto delle fee del **80%**, rendendo lo scalping sostenibile.

## Stack tecnico

| Componente | Libreria |
|---|---|
| Dati di mercato | `ccxt` + Binance API |
| Feature engineering | `pandas-ta` |
| Modello ML | `XGBClassifier` (xgboost) x2 |
| Ottimizzazione HP | `optuna` |
| Backtest | `backtesting.py` |
| Esecuzione ordini | `ccxt` + Binance Futures Demo |
| Dashboard | `flask` + Chart.js + Tailwind CSS |
| Notifiche | Telegram Bot API |

## Struttura file

```
CryptoBot/
  cryptobot.py          # bot scalping (dual model LONG+SHORT, Futures-only)
  dashboard.py          # dashboard web Flask (porta 5050)
  templates/
    dashboard.html      # template dashboard (dark theme, signal log)
  .env.example          # template per le credenziali
  requirements.txt      # dipendenze Python
  .gitignore
  CLAUDE.md             # istruzioni per Claude Code
  README.md
```

File generati a runtime (non committati):
- `model_buy_scalp.joblib` / `model_short_scalp.joblib` — modelli XGBoost
- `best_params_buy_scalp.json` / `best_params_short_scalp.json` — cache Optuna 48h
- `bot_state.json` — stato posizione + position_type
- `dashboard_data.json` — snapshot ciclo + signal_log
- `trades_log.csv` — log di tutti i trade
- `price_history.json` — ultime 100 candele per chart
- `MLStrategy.html` — plot interattivo del backtest

## Disclaimer

Questo bot e' un progetto educativo. Anche se opera su Binance Demo Trading (soldi finti), le performance passate non garantiscono risultati futuri. Non usare con soldi reali senza comprendere i rischi del trading algoritmico.

## License

MIT

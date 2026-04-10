# Graph Report - .  (2026-04-10)

## Corpus Check
- Corpus is ~13,927 words - fits in a single context window. You may not need a graph.

## Summary
- 131 nodes · 184 edges · 17 communities detected
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `run_bot()` - 16 edges
2. `cryptobot.py (Main Bot File)` - 16 edges
3. `run_bot()` - 13 edges
4. `retrain_model()` - 10 edges
5. `Python Dependencies (requirements.txt)` - 10 edges
6. `walk_forward_train()` - 8 edges
7. `walk_forward_train()` - 7 edges
8. `EnsembleClassifier` - 6 edges
9. `_train_single_model()` - 5 edges
10. `dashboard.py (Flask Dashboard)` - 5 edges

## Surprising Connections (you probably didn't know these)
- `walk_forward_train()` --calls--> `scikit-learn (train/test split, calibration)`  [INFERRED]
  CLAUDE.md → requirements.txt
- `CryptoBot Scalping README` --references--> `cryptobot.py (Main Bot File)`  [EXTRACTED]
  README.md → CLAUDE.md
- `CryptoBot Scalping README` --references--> `dashboard.py (Flask Dashboard)`  [EXTRACTED]
  README.md → CLAUDE.md
- `Project Context (CONTEXT.md)` --references--> `cryptobot.py (Main Bot File)`  [EXTRACTED]
  CONTEXT.md → CLAUDE.md
- `cryptobot.py (Main Bot File)` --implements--> `technical_sell_signal()`  [EXTRACTED]
  CLAUDE.md → CONTEXT.md

## Hyperedges (group relationships)
- **ML Training Pipeline** — fn_fetch_ohlcv, fn_build_features, fn_optimize_hyperparams, fn_walk_forward_train, concept_ensemble_classifier, concept_calibration [EXTRACTED 0.95]
- **Dual-Model Live Execution with Risk Management** — fn_run_bot, concept_dual_model, concept_circuit_breaker, concept_cooldown, concept_conflict_resolution, concept_exit_priority [EXTRACTED 0.92]
- **State Persistence and Recovery System** — fn_save_state, file_bot_state_json, concept_circuit_breaker, concept_cooldown, concept_bot_state [EXTRACTED 0.90]

## Communities

### Community 0 - "Risk and State Management"
Cohesion: 0.14
Nodes (18): Bot State Persistence (bot_state.json), Circuit Breaker (3 consecutive SL -> pause 2h), Conflict Resolution (BUY+SHORT -> HOLD), Cooldown Post-Exit, Entry Filters (trend_1h, trend_15m, adx_1h), Exit Priority Order (TP > Trail > SL > Gate > Tech > Hold), Futures-Only Rationale: Fee Impact on Scalping, Futures-Only Mode (+10 more)

### Community 1 - "Backtesting and Ensemble"
Cohesion: 0.15
Nodes (14): backtest(), EnsembleClassifier, Allena un XGBoost binary classifier.     target_label=1 per BUY vs NO-BUY, targ, Genera segnali di COPERTURA SHORT basati su regole tecniche (scalping):     - R, Ensemble di piu' modelli XGBoost con strategia max-con-agreement.     Compatibi, Max-con-agreement: se >= min_agree modelli concordano (proba >= agree_thresh),, Predizione basata sulla media delle probabilita' (soglia 0.5)., Allena un singolo XGBClassifier con guard per fold degeneri.     Ritorna (model (+6 more)

### Community 2 - "Feature Engineering"
Cohesion: 0.13
Nodes (16): 28 Scalping Features (5m+15m+1h), Backtest Performance Metrics (walk-forward LONG+SHORT), Multi-Timeframe Resampling (5m->15m->1h), Price Divisor (backtest integer sizing fix), backtest(), build_features(), fetch_ohlcv(), backtesting.py (Backtest Framework) (+8 more)

### Community 3 - "Documentation and Project Overview"
Cohesion: 0.16
Nodes (13): CryptoBot Project (CLAUDE.md), Dashboard API Endpoints (/api/status, trades, equity, candles), Project Context (CONTEXT.md), cryptobot.py (Main Bot File), dashboard.py (Flask Dashboard), bot_state.json (Runtime State File), dashboard_data.json (Dashboard Snapshot), save_dashboard_data() (+5 more)

### Community 4 - "ML Models and Calibration"
Cohesion: 0.23
Nodes (13): Backtest vs Live Trade Count Discrepancy, BUY Model (XGBClassifier), Ensemble Threshold Auto-Calibration, Degenerate Fold Fallback (previous model), Dual-Model LONG+SHORT, EnsembleClassifier (max-with-agreement), SHORT Model (XGBClassifier), shuffle=False (Time-Series Split Constraint) (+5 more)

### Community 5 - "Flask Dashboard"
Cohesion: 0.18
Nodes (9): candles(), equity(), CryptoBot Dashboard — Flask web app per monitorare il bot in tempo reale.  Avv, Ritorna lo snapshot corrente del bot., Ritorna gli ultimi 50 trade dal CSV., Calcola equity curve dai trade chiusi., Ritorna le ultime 100 candele OHLCV per il chart candlestick., status() (+1 more)

### Community 6 - "Trade Logging and PnL"
Cohesion: 0.25
Nodes (8): _calc_pnl(), log_trade(), Appende un trade al file CSV di log., Salva lo stato della posizione, circuit breaker e cooldown su file JSON., Calcola P&L correttamente per LONG e SHORT., Loop principale del bot scalping:     - Futures-only: LONG e SHORT entrambi su, run_bot(), save_state()

### Community 7 - "Data Fetch and Optimization"
Cohesion: 0.25
Nodes (8): fetch_ohlcv(), optimize_hyperparams(), Salva i modelli su disco con joblib (dual model: BUY + SHORT)., Scarica i dati OHLCV da Binance (endpoint pubblico, niente API key).     Suppor, Scarica dati freschi, rigenera feature, allena nuovi modelli (BUY + SHORT), Usa Optuna per trovare i migliori iperparametri XGBoost.     Valida su un split, retrain_model(), save_model()

### Community 8 - "Core Bot and Model Loading"
Cohesion: 0.29
Nodes (5): load_model(), load_state(), cryptobot.py ML-powered crypto trading bot for Binance Testnet. Fetches OHLCV, Carica lo stato della posizione, circuit breaker e cooldown da file JSON.     R, Carica i modelli da disco se esistono e hanno meno di RETRAIN_HOURS ore.     Ri

### Community 9 - "Hyperparameter Cache"
Cohesion: 0.4
Nodes (5): Optuna Hyperparameter Cache (48h TTL), best_params_buy_scalp.json (Optuna Cache), best_params_short_scalp.json (Optuna Cache), optimize_hyperparams(), optuna (Bayesian Hyperparameter Search)

### Community 10 - "Price History"
Cohesion: 1.0
Nodes (2): Salva le ultime 100 candele OHLCV per il chart candlestick della dashboard., save_price_history()

### Community 11 - "Dashboard Data Writer"
Cohesion: 1.0
Nodes (2): Salva snapshot del ciclo corrente per la dashboard web., save_dashboard_data()

### Community 12 - "Telegram Notifications"
Cohesion: 1.0
Nodes (2): Invia un messaggio tramite Telegram Bot API.     Non solleva eccezioni se il me, send_telegram()

### Community 13 - "Technical Exit Signals"
Cohesion: 1.0
Nodes (2): Genera segnali SELL basati su regole tecniche (scalping):     - RSI fast > 75 (, technical_sell_signal()

### Community 14 - "Feature Builder"
Cohesion: 1.0
Nodes (2): build_features(), Aggiunge indicatori tecnici scalping e calcola il label:       +1  = BUY  (prez

### Community 15 - "Exchange Connection"
Cohesion: 1.0
Nodes (2): get_testnet_exchange(), Ritorna un'istanza ccxt connessa al Binance Demo Trading.     Scalping: se USE_

### Community 16 - "Feature Importance"
Cohesion: 1.0
Nodes (1): Media delle feature importance di tutti i modelli.

## Knowledge Gaps
- **52 isolated node(s):** `cryptobot.py ML-powered crypto trading bot for Binance Testnet. Fetches OHLCV`, `Scarica i dati OHLCV da Binance (endpoint pubblico, niente API key).     Suppor`, `Aggiunge indicatori tecnici scalping e calcola il label:       +1  = BUY  (prez`, `Allena un XGBoost binary classifier.     target_label=1 per BUY vs NO-BUY, targ`, `Usa Optuna per trovare i migliori iperparametri XGBoost.     Valida su un split` (+47 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Price History`** (2 nodes): `Salva le ultime 100 candele OHLCV per il chart candlestick della dashboard.`, `save_price_history()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Dashboard Data Writer`** (2 nodes): `Salva snapshot del ciclo corrente per la dashboard web.`, `save_dashboard_data()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Telegram Notifications`** (2 nodes): `Invia un messaggio tramite Telegram Bot API.     Non solleva eccezioni se il me`, `send_telegram()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Technical Exit Signals`** (2 nodes): `Genera segnali SELL basati su regole tecniche (scalping):     - RSI fast > 75 (`, `technical_sell_signal()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Feature Builder`** (2 nodes): `build_features()`, `Aggiunge indicatori tecnici scalping e calcola il label:       +1  = BUY  (prez`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Exchange Connection`** (2 nodes): `get_testnet_exchange()`, `Ritorna un'istanza ccxt connessa al Binance Demo Trading.     Scalping: se USE_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Feature Importance`** (1 nodes): `Media delle feature importance di tutti i modelli.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `cryptobot.py (Main Bot File)` connect `Documentation and Project Overview` to `Risk and State Management`, `Hyperparameter Cache`, `Feature Engineering`, `ML Models and Calibration`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Why does `run_bot()` connect `Risk and State Management` to `Feature Engineering`, `Documentation and Project Overview`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `walk_forward_train()` connect `ML Models and Calibration` to `Feature Engineering`, `Documentation and Project Overview`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **What connects `cryptobot.py ML-powered crypto trading bot for Binance Testnet. Fetches OHLCV`, `Scarica i dati OHLCV da Binance (endpoint pubblico, niente API key).     Suppor`, `Aggiunge indicatori tecnici scalping e calcola il label:       +1  = BUY  (prez` to the rest of the system?**
  _52 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Risk and State Management` be split into smaller, more focused modules?**
  _Cohesion score 0.14 - nodes in this community are weakly interconnected._
- **Should `Feature Engineering` be split into smaller, more focused modules?**
  _Cohesion score 0.13 - nodes in this community are weakly interconnected._
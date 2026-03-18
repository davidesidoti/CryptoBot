"""
cryptobot.py
ML-powered crypto trading bot for Binance Testnet.
Fetches OHLCV data, engineers features, trains an XGBoost classifier,
backtests on historical data, and runs live paper trading on Binance Testnet.
"""

import os
import csv
import json
import time
from datetime import datetime, timezone
import warnings
import urllib.request
import ccxt
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from backtesting import Backtest, Strategy
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# Carica le variabili dal file .env (deve stare nella stessa cartella del bot)
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG -- edit these before running
# ─────────────────────────────────────────────

TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
TESTNET_SECRET  = os.getenv("BINANCE_TESTNET_SECRET", "")

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID", "")

if not TESTNET_API_KEY or not TESTNET_SECRET:
    print("[WARN] BINANCE_TESTNET_API_KEY o BINANCE_TESTNET_SECRET non trovati nel .env.")
    print("       Il bot funzionerà in modalità backtest, ma non potrà piazzare ordini live.")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
    print("[WARN] TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID non trovati nel .env.")
    print("       Le notifiche Telegram saranno disabilitate.")

SYMBOL          = "BTC/USDT"
TIMEFRAME       = "1h"        # candele orarie
FETCH_LIMIT     = 10000       # quante candele storiche scaricare (paginato, ~416 giorni)
N_TRAIN         = 400         # candele usate per il training
FUTURE_BARS     = 5           # quante candele in avanti per calcolare il label
SIGNAL_THRESH   = 0.007       # 0.7% di movimento minimo per generare segnale
MIN_PROBA       = 0.55        # confidenza minima per eseguire un ordine
TRADE_SIZE      = 0.95        # % del capitale usata per ogni trade
STOP_LOSS       = 0.02        # 2% stop loss — chiude posizione se la perdita supera questa soglia
INITIAL_CASH    = 500         # capitale iniziale per il backtest (in USD)
SLEEP_SECONDS   = 900         # secondi tra ogni ciclo del bot (15 min)
MAX_RETRIES     = 3           # tentativi per errori di rete transitori
RETRY_BACKOFF   = [30, 60, 120]  # secondi di attesa tra retry
MIN_HOLD_BARS   = 5           # hold minimo 5h prima di SELL tecnico (= FUTURE_BARS, stop loss escluso)
LOG_FILE        = "trades_log.csv"
RETRAIN_HOURS   = 24            # riaddestra il modello ogni N ore
MODEL_FILE      = "model.joblib"
STATE_FILE      = "bot_state.json"
DASHBOARD_FILE  = "dashboard_data.json"
PRICE_HISTORY_FILE = "price_history.json"

# SHORT parameters
SHORT_STOP_LOSS    = 0.02    # 2% stop loss per SHORT
SHORT_MIN_PROBA    = 0.60    # soglia piu' alta (SHORT piu' selettivo)
SHORT_TRADE_SIZE   = 0.70    # 70% capitale (meno aggressivo del LONG 95%)
SHORT_MIN_HOLD     = 3       # hold minimo 3h (bear moves piu' veloci)
ENABLE_SHORT       = True    # flag per abilitare/disabilitare SHORT

# Model files (dual model)
MODEL_BUY_FILE     = "model_buy.joblib"
MODEL_SHORT_FILE   = "model_short.joblib"

TRANSIENT_ERRORS = (
    ccxt.NetworkError,      # copre ExchangeNotAvailable, RequestTimeout, DDoSProtection
)

FEATURES = [
    "rsi", "macd", "macd_signal", "bb_width",
    "vol_change", "price_change", "ema_cross",
    "atr", "obv_change", "stoch_k", "rsi_slope", "hour",
    # Nuove features
    "adx", "willr", "vwap_dist",
    # Multi-timeframe features (resampled da 1h)
    "rsi_4h", "macd_4h", "ema_cross_4h", "trend_4h",
    "rsi_1d", "adx_1d",
    # Regime / volatilita'
    "atr_ratio", "vol_regime",
    # SHORT-specific
    "trend_down", "macd_hist"
]

# Walk-forward validation
WF_TRAIN_BARS   = 1500        # candele per ogni finestra di training
WF_TEST_BARS    = 200         # candele per ogni finestra di test

# ─────────────────────────────────────────────
# 1. FETCH DATI
# ─────────────────────────────────────────────

_ohlcv_exchange = None  # cache globale per evitare load_markets() ogni ciclo

def fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=FETCH_LIMIT):
    """
    Scarica i dati OHLCV da Binance (endpoint pubblico, niente API key).
    Supporta paginazione per ottenere piu' di 1000 candele.
    Usa istanza cached per evitare chiamate ripetute a exchangeInfo.
    Fallback: se l'API pubblica fallisce, usa il demo endpoint.
    """
    global _ohlcv_exchange
    if _ohlcv_exchange is None:
        try:
            _ohlcv_exchange = ccxt.binance({
                "options": {"fetchMarkets": {"types": ["spot"]}},
            })
            _ohlcv_exchange.load_markets()
        except Exception:
            # Fallback: usa demo endpoint (stessi dati di mercato)
            print("[WARN] API pubblica Binance non raggiungibile, uso demo endpoint")
            _ohlcv_exchange = ccxt.binance({
                "apiKey": TESTNET_API_KEY,
                "secret": TESTNET_SECRET,
                "options": {"defaultType": "spot", "fetchMarkets": {"types": ["spot"]}},
            })
            _ohlcv_exchange.enable_demo_trading(True)
            _ohlcv_exchange.load_markets()
    exchange = _ohlcv_exchange
    max_per_request = 1000

    if limit <= max_per_request:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    else:
        # Paginazione: scarica a blocchi da 1000 andando indietro nel tempo
        all_ohlcv = []
        remaining = limit
        since = None

        # Prima richiesta: ultime 1000 candele
        batch = exchange.fetch_ohlcv(symbol, timeframe, limit=max_per_request)
        all_ohlcv = batch
        remaining -= len(batch)

        # Richieste successive: va indietro nel tempo
        while remaining > 0 and len(batch) == max_per_request:
            earliest_ts = all_ohlcv[0][0]
            # Calcola il timestamp da cui partire (prima della candela piu' vecchia)
            tf_ms = exchange.parse_timeframe(timeframe) * 1000
            since = earliest_ts - max_per_request * tf_ms
            batch_size = min(remaining, max_per_request)
            batch = exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=batch_size
            )
            if not batch:
                break
            # Filtra candele gia' presenti
            batch = [c for c in batch if c[0] < earliest_ts]
            all_ohlcv = batch + all_ohlcv
            remaining -= len(batch)

        ohlcv = all_ohlcv[-limit:]  # tronca al numero richiesto

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING + LABEL
# ─────────────────────────────────────────────

def build_features(df):
    """
    Aggiunge indicatori tecnici e calcola il label:
      +1  = BUY  (prezzo sale > soglia dinamica ATR nelle prossime FUTURE_BARS candele)
      -1  = SELL (prezzo scende)
       0  = HOLD
    Include feature multi-timeframe (4h, 1d) resamplandole dal dataset 1h.
    """
    df = df.copy()

    # === Feature 1h (base) ===
    df["rsi"]         = ta.rsi(df["close"], length=14)
    macd_df           = ta.macd(df["close"])
    df["macd"]        = macd_df["MACD_12_26_9"]
    df["macd_signal"] = macd_df["MACDs_12_26_9"]
    bb                = ta.bbands(df["close"], length=20)
    df["bb_upper"]    = bb["BBU_20_2.0_2.0"]
    df["bb_lower"]    = bb["BBL_20_2.0_2.0"]
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    df["vol_change"]  = df["volume"].pct_change()
    df["price_change"]= df["close"].pct_change()
    df["ema_9"]       = ta.ema(df["close"], length=9)
    df["ema_21"]      = ta.ema(df["close"], length=21)
    df["ema_cross"]   = df["ema_9"] - df["ema_21"]
    df["atr"]         = ta.atr(df["high"], df["low"], df["close"], length=14)
    obv               = ta.obv(df["close"], df["volume"])
    df["obv_change"]  = obv.pct_change()
    stoch              = ta.stoch(df["high"], df["low"], df["close"])
    df["stoch_k"]     = stoch.iloc[:, 0]
    df["rsi_slope"]   = df["rsi"].diff(5)
    df["hour"]        = df.index.hour
    df["ema_20"]      = ta.ema(df["close"], length=20)
    df["trend_up"]    = (df["close"] > df["ema_20"]).astype(int)

    # Nuove features
    adx_df            = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx"]         = adx_df["ADX_14"]          # forza del trend (0-100)
    df["willr"]       = ta.willr(df["high"], df["low"], df["close"], length=14)  # Williams %R
    # VWAP distance: distanza % del prezzo dal VWAP rolling (proxy intraday)
    cum_vol           = df["volume"].rolling(20).sum()
    cum_vp            = (df["close"] * df["volume"]).rolling(20).sum()
    vwap_20           = cum_vp / cum_vol
    df["vwap_dist"]   = (df["close"] - vwap_20) / vwap_20  # >0 = sopra VWAP

    # === Feature multi-timeframe (resample da 1h) ===
    # 4h
    df_4h = df[["open", "high", "low", "close", "volume"]].resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    df_4h["rsi_4h"]      = ta.rsi(df_4h["close"], length=14)
    macd_4h              = ta.macd(df_4h["close"])
    df_4h["macd_4h"]     = macd_4h["MACD_12_26_9"]
    ema9_4h              = ta.ema(df_4h["close"], length=9)
    ema21_4h             = ta.ema(df_4h["close"], length=21)
    df_4h["ema_cross_4h"] = ema9_4h - ema21_4h
    ema20_4h             = ta.ema(df_4h["close"], length=20)
    df_4h["trend_4h"]    = (df_4h["close"] > ema20_4h).astype(int)
    # Forward-fill su timeframe 1h (ogni candela 4h vale per le 4 candele 1h successive)
    for col in ["rsi_4h", "macd_4h", "ema_cross_4h", "trend_4h"]:
        df[col] = df_4h[col].reindex(df.index, method="ffill")

    # 1d
    df_1d = df[["open", "high", "low", "close", "volume"]].resample("1D").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    df_1d["rsi_1d"]   = ta.rsi(df_1d["close"], length=14)
    adx_1d_df         = ta.adx(df_1d["high"], df_1d["low"], df_1d["close"], length=14)
    df_1d["adx_1d"]   = adx_1d_df["ADX_14"]  # forza trend daily (sostituisce trend_1d)
    for col in ["rsi_1d", "adx_1d"]:
        df[col] = df_1d[col].reindex(df.index, method="ffill")

    # === Regime / volatilita' ===
    atr_fast = ta.atr(df["high"], df["low"], df["close"], length=7)
    atr_slow = ta.atr(df["high"], df["low"], df["close"], length=28)
    df["atr_ratio"]  = atr_fast / atr_slow  # >1 = volatilita' in aumento
    vol_20 = df["volume"].rolling(20).mean()
    df["vol_regime"] = df["volume"] / vol_20  # >1 = volume sopra media

    # === SHORT-specific features ===
    df["trend_down"] = (df["close"] < df["ema_20"]).astype(int)  # inverso di trend_up
    df["macd_hist"]  = df["macd"] - df["macd_signal"]            # MACD histogram esplicito

    # === Target dinamico (soglia basata su ATR) ===
    # Usa ATR% come soglia: se mercato volatile, servono movimenti piu' grandi
    atr_pct = df["atr"] / df["close"]
    # La soglia e' il massimo tra SIGNAL_THRESH fisso e 0.5*ATR%
    # Questo evita falsi segnali in mercati volatili
    dynamic_thresh = np.maximum(SIGNAL_THRESH, atr_pct * 0.5)

    future_return     = df["close"].shift(-FUTURE_BARS) / df["close"] - 1
    df["label"]       = 0
    df.loc[future_return >  dynamic_thresh, "label"] =  1
    df.loc[future_return < -dynamic_thresh, "label"] = -1

    return df.dropna()


# ─────────────────────────────────────────────
# 3. TRAINING MODELLO
# ─────────────────────────────────────────────

def train_model(df, target_label=1):
    """
    Allena un XGBoost binary classifier.
    target_label=1 per BUY vs NO-BUY, target_label=-1 per SHORT vs NO-SHORT.
    Usa shuffle=False perche' i dati sono time-series.
    """
    label_name = "BUY" if target_label == 1 else "SHORT"
    neg_name = "NO-BUY" if target_label == 1 else "NO-SHORT"

    X = df[FEATURES]
    y = (df["label"] == target_label).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Diagnostica distribuzione classi
    print(f"Distribuzione classi {label_name} (train):")
    for cls in [0, 1]:
        count = (y_train == cls).sum()
        name = label_name if cls == 1 else neg_name
        print(f"  {name}: {count} ({count/len(y_train):.1%})")

    # scale_pos_weight: bilancia automaticamente la classe rara
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / pos_count if pos_count > 0 else 1.0

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=1.5,
        scale_pos_weight=spw,
        eval_metric="logloss",
        verbosity=0,
        early_stopping_rounds=80
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print(f"\nMiglior iterazione ({label_name}): {model.best_iteration} / 500")
    print(f"\n=== Valutazione modello {label_name} sul test set ===")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=[neg_name, label_name]))

    return model


# ─────────────────────────────────────────────
# 3b. OTTIMIZZAZIONE IPERPARAMETRI (OPTUNA)
# ─────────────────────────────────────────────

OPTUNA_BUY_FILE   = "best_params_buy.json"
OPTUNA_SHORT_FILE = "best_params_short.json"
OPTUNA_TRIALS     = 50  # numero di trial per la ricerca


def optimize_hyperparams(df, target_label=1, cache_file=None, n_trials=OPTUNA_TRIALS):
    """
    Usa Optuna per trovare i migliori iperparametri XGBoost.
    Valida su un split temporale (ultimi 20% del dataset).
    target_label: 1 per BUY, -1 per SHORT.
    cache_file: file JSON per caching (default: OPTUNA_BUY_FILE o OPTUNA_SHORT_FILE).
    """
    if cache_file is None:
        cache_file = OPTUNA_BUY_FILE if target_label == 1 else OPTUNA_SHORT_FILE

    label_name = "BUY" if target_label == 1 else "SHORT"

    # Se esiste un file recente (< 48h), riusa i parametri
    if os.path.isfile(cache_file):
        age_h = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_h < 48:
            with open(cache_file) as f:
                params = json.load(f)
            print(f"Iperparametri {label_name} caricati da {cache_file} (eta': {age_h:.1f}h)")
            return params

    print(f"\n=== Ottimizzazione iperparametri {label_name} ({n_trials} trial) ===")
    X = df[FEATURES]
    y = (df["label"] == target_label).astype(int)

    # Split temporale: train 70%, val 15%, test 15% (test non usato qui)
    split_1 = int(len(X) * 0.70)
    split_2 = int(len(X) * 0.85)
    X_train, y_train = X.iloc[:split_1], y.iloc[:split_1]
    X_val, y_val = X.iloc[split_1:split_2], y.iloc[split_1:split_2]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = min(neg / pos, 50.0) if pos > 0 else 1.0
    print(f"  Samples positivi ({label_name}): {pos}, negativi: {neg}, scale_pos_weight: {spw:.1f}")

    def objective(trial):
        params = {
            "n_estimators": 500,
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            "scale_pos_weight": spw,
            "eval_metric": "logloss",
            "verbosity": 0,
            "early_stopping_rounds": 80,
        }
        m = XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        preds = m.predict(X_val)
        from sklearn.metrics import f1_score
        return f1_score(y_val, preds, pos_label=1, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"Migliori parametri {label_name} (F1: {study.best_value:.3f}):")
    for k, v in best.items():
        print(f"  {k}: {v}")

    # Salva su disco
    with open(cache_file, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Salvati in {cache_file}")

    return best


# ─────────────────────────────────────────────
# 3c. WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────

def technical_sell_signal(df):
    """
    Genera segnali SELL basati su regole tecniche (non ML):
    - RSI > 70 (ipercomprato)
    - MACD cross ribassista (MACD < signal e in discesa)
    - Prezzo sotto EMA20 (trend ribassista)
    Ritorna una Series booleana con True dove c'e' segnale SELL.
    """
    sell = pd.Series(False, index=df.index)

    # RSI ipercomprato + inizio discesa (75 = vero ipercomprato per BTC)
    sell |= (df["rsi"] > 75) & (df["rsi_slope"] < 0)

    # MACD cross ribassista confermato (gap in aumento per 2 barre consecutive)
    macd_gap = df["macd"] - df["macd_signal"]
    sell |= (macd_gap < 0) & (macd_gap.diff() < 0) & (macd_gap.shift(1).diff() < 0)

    # Prezzo crolla sotto EMA20 con momentum significativo (0.8%)
    sell |= (df["trend_up"] == 0) & (df["price_change"] < -0.008)

    return sell


def technical_cover_signal(df):
    """
    Genera segnali di COPERTURA SHORT basati su regole tecniche (speculare a sell):
    - RSI < 25 (ipervenduto) + rimbalzo (rsi_slope > 0)
    - MACD cross rialzista confermato (gap in aumento per 2 barre consecutive)
    - Prezzo risale sopra EMA20 con momentum significativo (0.8%)
    Ritorna una Series booleana con True dove c'e' segnale COVER (chiudi SHORT).
    """
    cover = pd.Series(False, index=df.index)

    # RSI ipervenduto + inizio rimbalzo
    cover |= (df["rsi"] < 25) & (df["rsi_slope"] > 0)

    # MACD cross rialzista confermato (gap in aumento per 2 barre consecutive)
    macd_gap = df["macd"] - df["macd_signal"]
    cover |= (macd_gap > 0) & (macd_gap.diff() > 0) & (macd_gap.shift(1).diff() > 0)

    # Prezzo risale sopra EMA20 con momentum significativo (0.8%)
    cover |= (df["trend_up"] == 1) & (df["price_change"] > 0.008)

    return cover


def _train_single_model(X_train, y_train, X_test, y_test, hp, prev_model, min_samples=20):
    """
    Allena un singolo XGBClassifier con guard per fold degeneri.
    Ritorna (model_or_fallback, preds, proba, is_degenerate, prev_model).
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = min(neg_count / pos_count, 50.0) if pos_count > 0 else 1.0

    mdl = XGBClassifier(
        n_estimators=500,
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        reg_alpha=hp["reg_alpha"],
        reg_lambda=hp["reg_lambda"],
        scale_pos_weight=spw,
        eval_metric="logloss",
        verbosity=0,
        early_stopping_rounds=80
    )
    mdl.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    is_degenerate = mdl.best_iteration < 10 or pos_count < min_samples
    if is_degenerate and prev_model is not None:
        preds = prev_model.predict(X_test)
        proba = prev_model.predict_proba(X_test)[:, 1]
    elif is_degenerate:
        preds = np.zeros(len(X_test), dtype=int)
        proba = np.full(len(X_test), 0.0)
    else:
        preds = mdl.predict(X_test)
        proba = mdl.predict_proba(X_test)[:, 1]
        prev_model = mdl

    return mdl, preds, proba, is_degenerate, prev_model


def walk_forward_train(df, best_params_buy=None, best_params_short=None):
    """
    Walk-forward validation con dual binary classification:
    - model_buy:   BUY (1) vs NO-BUY (0)
    - model_short: SHORT (1) vs NO-SHORT (0)
    Il segnale CLOSE LONG viene da regole tecniche (technical_sell_signal).
    Il segnale CLOSE SHORT viene da regole tecniche (technical_cover_signal).
    Per ogni finestra, genera predictions out-of-sample.
    Signal encoding: +1=OPEN LONG, -1=OPEN SHORT, -2=CLOSE LONG, +2=CLOSE SHORT, 0=HOLD.
    Se ENABLE_SHORT=False, il model_short non viene trainato.
    """
    X = df[FEATURES]
    y_buy   = (df["label"] == 1).astype(int)
    y_short = (df["label"] == -1).astype(int)

    # Risultati per fold (BUY)
    all_buy_preds  = []
    all_buy_proba  = []
    all_buy_actual = []
    # Risultati per fold (SHORT)
    all_short_preds  = []
    all_short_proba  = []
    all_short_actual = []
    all_idx    = []
    n = len(df)
    fold = 0
    prev_buy_model = None
    prev_short_model = None
    MIN_SAMPLES = 20

    # Iperparametri BUY: Optuna o default
    hp_buy = {
        "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 10,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 0.3, "reg_lambda": 1.5,
    }
    if best_params_buy:
        hp_buy.update(best_params_buy)
        print(f"Usando iperparametri BUY ottimizzati (Optuna)")

    # Iperparametri SHORT: Optuna o default
    hp_short = {
        "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 10,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 0.3, "reg_lambda": 1.5,
    }
    if best_params_short:
        hp_short.update(best_params_short)
        print(f"Usando iperparametri SHORT ottimizzati (Optuna)")

    short_enabled = ENABLE_SHORT
    mode_str = "LONG+SHORT" if short_enabled else "LONG-only"
    print(f"Walk-forward ({mode_str}): train={WF_TRAIN_BARS}, test={WF_TEST_BARS}, "
          f"dataset={n} righe")

    i = 0
    while i + WF_TRAIN_BARS + WF_TEST_BARS <= n:
        fold += 1
        train_end = i + WF_TRAIN_BARS
        test_end  = min(train_end + WF_TEST_BARS, n)

        X_train = X.iloc[i:train_end]
        X_test  = X.iloc[train_end:test_end]

        # --- BUY model ---
        y_buy_train = y_buy.iloc[i:train_end]
        y_buy_test  = y_buy.iloc[train_end:test_end]
        buy_mdl, buy_preds, buy_proba, buy_degen, prev_buy_model = _train_single_model(
            X_train, y_buy_train, X_test, y_buy_test, hp_buy, prev_buy_model, MIN_SAMPLES
        )

        # --- SHORT model ---
        if short_enabled:
            y_short_train = y_short.iloc[i:train_end]
            y_short_test  = y_short.iloc[train_end:test_end]
            short_mdl, short_preds, short_proba, short_degen, prev_short_model = _train_single_model(
                X_train, y_short_train, X_test, y_short_test, hp_short, prev_short_model, MIN_SAMPLES
            )
        else:
            short_preds = np.zeros(len(X_test), dtype=int)
            short_proba = np.full(len(X_test), 0.0)
            short_degen = True

        all_buy_preds.extend(buy_preds)
        all_buy_proba.extend(buy_proba)
        all_buy_actual.extend(y_buy_test.values)
        all_short_preds.extend(short_preds)
        all_short_proba.extend(short_proba)
        all_short_actual.extend(y_short.iloc[train_end:test_end].values)
        all_idx.extend(y_buy_test.index)

        buy_cnt = sum(buy_preds)
        short_cnt = sum(short_preds) if short_enabled else 0
        buy_status = " [DEGEN]" if buy_degen else ""
        short_status = " [DEGEN]" if short_degen else ""
        print(f"  Fold {fold}: train [{i}:{train_end}] | "
              f"test [{train_end}:{test_end}] | "
              f"BUY: {buy_cnt}/{len(buy_preds)}{buy_status} | "
              f"SHORT: {short_cnt}/{len(short_preds)}{short_status}")

        i += WF_TEST_BARS

    print(f"\n=== Walk-forward: {fold} fold completati ({mode_str}) ===")
    print(f"Predictions out-of-sample totali: {len(all_buy_preds)}")

    # Classification report BUY
    print("\n=== Classification report BUY (aggregato) ===")
    print(classification_report(
        all_buy_actual, all_buy_preds,
        target_names=["NO-BUY", "BUY"]
    ))

    # Classification report SHORT
    if short_enabled:
        print("=== Classification report SHORT (aggregato) ===")
        print(classification_report(
            all_short_actual, all_short_preds,
            target_names=["NO-SHORT", "SHORT"]
        ))

    # Analisi qualita' segnali BUY
    buy_preds_arr = np.array(all_buy_preds)
    buy_actual_arr = np.array(all_buy_actual)
    if buy_preds_arr.sum() > 0:
        buy_precision = buy_actual_arr[buy_preds_arr == 1].mean()
        print(f"BUY precision effettiva: {buy_precision:.1%} "
              f"({buy_preds_arr.sum()} segnali BUY predetti)")

    # Analisi qualita' segnali SHORT
    if short_enabled:
        short_preds_arr = np.array(all_short_preds)
        short_actual_arr = np.array(all_short_actual)
        if short_preds_arr.sum() > 0:
            short_precision = short_actual_arr[short_preds_arr == 1].mean()
            print(f"SHORT precision effettiva: {short_precision:.1%} "
                  f"({short_preds_arr.sum()} segnali SHORT predetti)")

    # === Genera segnali combinati per il backtest ===
    # Signal encoding:
    #   +1 = OPEN LONG   (ML BUY con alta confidenza)
    #   -1 = OPEN SHORT  (ML SHORT con alta confidenza)
    #   -2 = CLOSE LONG  (technical sell signal)
    #   +2 = CLOSE SHORT (technical cover signal)
    #    0 = HOLD
    sell_signals  = technical_sell_signal(df)
    cover_signals = technical_cover_signal(df)

    pred_values = np.zeros(len(all_buy_preds))
    n_conflicts = 0
    for j in range(len(all_buy_preds)):
        idx = all_idx[j]
        buy_ok   = all_buy_preds[j] == 1 and all_buy_proba[j] >= MIN_PROBA
        short_ok = short_enabled and all_short_preds[j] == 1 and all_short_proba[j] >= SHORT_MIN_PROBA

        if buy_ok and short_ok:
            pred_values[j] = 0  # conflitto: entrambi i modelli "sparano" -> HOLD
            n_conflicts += 1
        elif buy_ok:
            pred_values[j] = 1   # OPEN LONG
        elif short_ok:
            pred_values[j] = -1  # OPEN SHORT
        elif sell_signals.loc[idx]:
            pred_values[j] = -2  # CLOSE LONG (technical)
        elif short_enabled and cover_signals.loc[idx]:
            pred_values[j] = 2   # CLOSE SHORT (technical)
        # else: 0 (HOLD)

    pred_series = pd.Series(pred_values.astype(int), index=all_idx)

    # Stats dei segnali
    n_open_long  = (pred_values == 1).sum()
    n_open_short = (pred_values == -1).sum()
    n_close_long = (pred_values == -2).sum()
    n_close_short = (pred_values == 2).sum()
    print(f"\nSegnali generati: OPEN_LONG={n_open_long}, OPEN_SHORT={n_open_short}, "
          f"CLOSE_LONG={n_close_long}, CLOSE_SHORT={n_close_short}, "
          f"HOLD={int((pred_values == 0).sum())}, CONFLITTI={n_conflicts}")

    # Ritorna entrambi i modelli dell'ultimo fold valido
    final_buy_model = prev_buy_model if prev_buy_model else buy_mdl
    final_short_model = prev_short_model if (short_enabled and prev_short_model) else (short_mdl if short_enabled else None)

    return final_buy_model, final_short_model, pred_series


# ─────────────────────────────────────────────
# 4. BACKTEST
# ─────────────────────────────────────────────

def backtest(df_raw, df_feat, model_buy, model_short=None, pred_series=None, test_size=0.2):
    """
    Backtesta la strategia ML solo sui dati out-of-sample (test set)
    per evitare data leakage. Include stop loss e report P&L in dollari.
    Se pred_series e' fornita (da walk-forward), usa quelle predictions.
    Supporta LONG + SHORT con signal encoding:
      +1=OPEN LONG, -1=OPEN SHORT, -2=CLOSE LONG, +2=CLOSE SHORT, 0=HOLD.
    """

    if pred_series is not None:
        # Walk-forward: usa le predictions aggregate out-of-sample
        feat_index  = pred_series.index
        raw_aligned = df_raw.loc[feat_index].copy()
    else:
        # Singolo split: genera predictions dal modello (LONG-only fallback)
        split_idx   = int(len(df_feat) * (1 - test_size))
        df_test     = df_feat.iloc[split_idx:]
        feat_index  = df_test.index
        raw_aligned = df_raw.loc[feat_index].copy()
        predictions = model_buy.predict(df_test[FEATURES]) - 1
        pred_series = pd.Series(predictions, index=feat_index)

    # Filtro trend: blocca BUY quando prezzo < EMA20, blocca SHORT quando prezzo > EMA20
    trend_up = df_feat.loc[feat_index, "trend_up"]

    class MLStrategy(Strategy):
        def init(self):
            self.signal = self.I(
                lambda: pred_series.reindex(raw_aligned.index).fillna(0).values,
                name="ML Signal"
            )
            self.trend = self.I(
                lambda: trend_up.reindex(raw_aligned.index).fillna(0).values,
                name="Trend"
            )
            self.entry_price = None
            self.bars_held = 0

        def next(self):
            sig = self.signal[-1]
            price = self.data.Close[-1]
            trend_ok = self.trend[-1] == 1

            # Stop loss LONG: chiudi se la perdita supera STOP_LOSS (SEMPRE attivo)
            if self.position.is_long and self.entry_price:
                self.bars_held += 1
                loss = (price - self.entry_price) / self.entry_price
                if loss <= -STOP_LOSS:
                    self.position.close()
                    self._reset()
                    return

            # Stop loss SHORT: chiudi se prezzo sale oltre SHORT_STOP_LOSS (SEMPRE attivo)
            if self.position.is_short and self.entry_price:
                self.bars_held += 1
                loss = (self.entry_price - price) / self.entry_price  # invertito
                if loss <= -SHORT_STOP_LOSS:
                    self.position.close()
                    self._reset()
                    return

            # OPEN LONG: sig == +1, flat, trend up
            if sig == 1 and not self.position and trend_ok:
                self.buy(size=TRADE_SIZE)
                self.entry_price = price
                self.bars_held = 0

            # OPEN SHORT: sig == -1, flat, trend down
            elif sig == -1 and not self.position and not trend_ok:
                self.sell(size=SHORT_TRADE_SIZE)
                self.entry_price = price
                self.bars_held = 0

            # CLOSE LONG: sig == -2, hold minimo rispettato
            elif sig == -2 and self.position.is_long:
                if self.bars_held < MIN_HOLD_BARS:
                    return
                self.position.close()
                self._reset()

            # CLOSE SHORT: sig == +2, hold minimo rispettato
            elif sig == 2 and self.position.is_short:
                if self.bars_held < SHORT_MIN_HOLD:
                    return
                self.position.close()
                self._reset()

        def _reset(self):
            self.entry_price = None
            self.bars_held = 0

    # backtesting.py richiede colonne con la prima lettera maiuscola
    bt_df = raw_aligned.rename(columns={
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume"
    })

    # backtesting.py arrotonda le posizioni a interi.
    # Con capitali piccoli e BTC a prezzi alti (es. $500 vs $80K),
    # l'ordine risulterebbe 0 unita'. Scaliamo i prezzi in modo che
    # il capitale possa comprare almeno ~10 unita'.
    # Il P&L in $ e i rendimenti % restano identici.
    last_price = bt_df["Close"].iloc[-1]
    price_divisor = max(1, int(last_price / (INITIAL_CASH * 0.1)))
    if price_divisor > 1:
        print(f"  [INFO] Prezzi scalati /{price_divisor} per backtest "
              f"(capitale piccolo vs prezzo asset)")
        for col in ["Open", "High", "Low", "Close"]:
            bt_df[col] = bt_df[col] / price_divisor

    bt = Backtest(
        bt_df,
        MLStrategy,
        cash=INITIAL_CASH,
        commission=0.001,      # 0.1% commission, uguale a Binance
        exclusive_orders=True
    )
    stats = bt.run()

    print("\n=== Risultati Backtest ===")
    print(stats)

    # Report P&L in dollari
    n_trades     = stats["# Trades"]
    equity_final = stats["Equity Final [$]"]
    pnl_dollar   = equity_final - INITIAL_CASH

    print(f"\n{'=' * 45}")
    print(f"  RIEPILOGO P&L IN DOLLARI")
    print(f"{'=' * 45}")
    print(f"  Capitale iniziale:   ${INITIAL_CASH:>12,.2f}")
    print(f"  Capitale finale:     ${equity_final:>12,.2f}")

    if n_trades > 0:
        commissions = stats.get("Commissions [$]", 0)
        pnl_gross   = pnl_dollar + commissions
        best_pct    = stats["Best Trade [%]"]
        worst_pct   = stats["Worst Trade [%]"]
        print(f"  P&L lordo:           ${pnl_gross:>12,.2f}")
        print(f"  Commissioni:        -${commissions:>12,.2f}")
        print(f"  P&L netto:           ${pnl_dollar:>+12,.2f}")
        print(f"  Trade totali:         {n_trades:>11}")
        print(f"  Miglior trade:       ~${INITIAL_CASH * best_pct / 100:>+11,.2f} ({best_pct:+.2f}%)")
        print(f"  Peggior trade:       ~${INITIAL_CASH * worst_pct / 100:>+11,.2f} ({worst_pct:+.2f}%)")
    else:
        print(f"  P&L netto:           ${pnl_dollar:>+12,.2f}")
        print(f"  Trade totali:         {n_trades:>11}")
        print(f"  [WARN] Nessun trade eseguito.")

    print(f"{'=' * 45}")

    bt.plot()

    return stats


# ─────────────────────────────────────────────
# 5. PAPER TRADING LIVE (Binance Testnet)
# ─────────────────────────────────────────────

def get_testnet_exchange(exchange_type="spot"):
    """
    Ritorna un'istanza ccxt connessa al Binance Demo Trading.
    exchange_type: "spot" per LONG, "future" per SHORT.
    Usa enable_demo_trading() che punta a:
    - demo-api.binance.com (spot)
    - demo-fapi.binance.com (futures)
    """
    if exchange_type == "future":
        exchange = ccxt.binance({
            "apiKey": TESTNET_API_KEY,
            "secret": TESTNET_SECRET,
            "options": {
                "defaultType": "future",
            },
        })
        exchange.enable_demo_trading(True)
        exchange.load_markets()
        # Leva 1x: nessun leverage, puro short senza margine amplificato
        try:
            exchange.set_leverage(1, SYMBOL)
            print(f"Futures: leverage impostato a 1x per {SYMBOL}")
        except Exception as e:
            raise RuntimeError(
                f"Impossibile impostare leverage 1x su Futures: {e}. "
                f"Bot non avviato per sicurezza (rischio leverage non controllato)."
            )
        return exchange
    else:
        exchange = ccxt.binance({
            "apiKey": TESTNET_API_KEY,
            "secret": TESTNET_SECRET,
            "options": {
                "defaultType": "spot",
                "fetchMarkets": {"types": ["spot"]},
            },
        })
        exchange.enable_demo_trading(True)
        exchange.load_markets()
        return exchange


def log_trade(side, qty, price, signal, confidence, pnl_usd=None):
    """
    Appende un trade al file CSV di log.
    """
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "symbol", "side", "qty",
                "price", "signal", "confidence", "pnl_usd"
            ])
        writer.writerow([
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            SYMBOL, side, qty, f"{price:.2f}",
            signal, f"{confidence:.4f}",
            f"{pnl_usd:.2f}" if pnl_usd is not None else ""
        ])


# ─────────────────────────────────────────────
# 5b. NOTIFICHE TELEGRAM
# ─────────────────────────────────────────────

def send_telegram(message):
    """
    Invia un messaggio tramite Telegram Bot API.
    Non solleva eccezioni se il messaggio non viene inviato.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        url = (
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            f"?chat_id={TELEGRAM_CHAT}"
            f"&parse_mode=HTML"
            f"&text={urllib.request.quote(message)}"
        )
        urllib.request.urlopen(url, timeout=10)
    except Exception:
        pass  # non bloccare il bot per errori Telegram


# ─────────────────────────────────────────────
# 5c. PERSISTENZA STATO E MODELLO
# ─────────────────────────────────────────────

def save_state(entry_price, entry_qty, entry_time=None, position_type=None):
    """Salva lo stato della posizione su file JSON (LONG/SHORT/flat)."""
    with open(STATE_FILE, "w") as f:
        json.dump({
            "entry_price": entry_price,
            "entry_qty": entry_qty,
            "entry_time": entry_time.isoformat() if entry_time else None,
            "position_type": position_type,  # "long", "short", or None
        }, f)


def load_state():
    """
    Carica lo stato della posizione da file JSON.
    Ritorna (entry_price, entry_qty, entry_time, position_type).
    Backward compat: vecchi file senza position_type → assume "long" se entry_price esiste.
    """
    if os.path.isfile(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
            et = s.get("entry_time")
            entry_time = datetime.fromisoformat(et) if et else None
            ep = s.get("entry_price")
            # Backward compat: se entry_price esiste ma position_type manca, assume "long"
            pt = s.get("position_type")
            if ep and not pt:
                pt = "long"
            return ep, s.get("entry_qty"), entry_time, pt
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return None, None, None, None


def save_dashboard_data(price, buy_proba, signal_str, usdt, btc,
                        entry_price, entry_qty, features_row):
    """Salva snapshot del ciclo corrente per la dashboard web."""
    # Accumula ultimi 20 prezzi per sparkline
    sparkline = []
    if os.path.isfile(DASHBOARD_FILE):
        try:
            with open(DASHBOARD_FILE) as f:
                old = json.load(f)
            sparkline = old.get("sparkline", [])
        except (json.JSONDecodeError, KeyError):
            pass
    sparkline.append(round(price, 2))
    sparkline = sparkline[-20:]

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": price,
        "buy_proba": round(buy_proba, 4),
        "signal": signal_str,
        "usdt": round(usdt, 2),
        "btc": round(btc, 8),
        "entry_price": entry_price,
        "entry_qty": entry_qty,
        "pnl_pct": round((price - entry_price) / entry_price, 4) if entry_price else None,
        "pnl_usd": round((price - entry_price) * (entry_qty or 0), 2) if entry_price else None,
        "features": {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in features_row.items()},
        "sparkline": sparkline,
        "sleep_seconds": SLEEP_SECONDS,
        "stop_loss": STOP_LOSS,
        "min_proba": MIN_PROBA,
    }
    with open(DASHBOARD_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_price_history(df):
    """Salva le ultime 100 candele OHLCV per il chart candlestick della dashboard."""
    try:
        recent = df.tail(100)[["open", "high", "low", "close", "volume"]].copy()
        recent["timestamp"] = recent.index.strftime("%Y-%m-%dT%H:%M:%S")
        with open(PRICE_HISTORY_FILE, "w") as f:
            json.dump(recent.to_dict(orient="records"), f)
    except Exception as e:
        print(f"[WARN] save_price_history fallito: {e}")


def save_model(model_buy, model_short=None):
    """Salva i modelli su disco con joblib (dual model: BUY + SHORT)."""
    joblib.dump(model_buy, MODEL_BUY_FILE)
    print(f"Modello BUY salvato in {MODEL_BUY_FILE}")
    if model_short is not None:
        joblib.dump(model_short, MODEL_SHORT_FILE)
        print(f"Modello SHORT salvato in {MODEL_SHORT_FILE}")


def load_model():
    """
    Carica i modelli da disco se esistono e hanno meno di RETRAIN_HOURS ore.
    Ritorna (model_buy, model_short) o (None, None).
    Backward compat: se esiste il vecchio model.joblib, lo usa come model_buy.
    """
    model_buy = None
    model_short = None

    # Prova prima i nuovi file dual
    if os.path.isfile(MODEL_BUY_FILE):
        age_hours = (time.time() - os.path.getmtime(MODEL_BUY_FILE)) / 3600
        if age_hours > RETRAIN_HOURS:
            print(f"Modello BUY troppo vecchio ({age_hours:.1f}h). Ri-training necessario.")
            return None, None
        model_buy = joblib.load(MODEL_BUY_FILE)
        print(f"Modello BUY caricato da {MODEL_BUY_FILE} (eta': {age_hours:.1f}h)")
    elif os.path.isfile(MODEL_FILE):
        # Backward compat: vecchio file singolo
        age_hours = (time.time() - os.path.getmtime(MODEL_FILE)) / 3600
        if age_hours > RETRAIN_HOURS:
            print(f"Modello trovato ma troppo vecchio ({age_hours:.1f}h). Ri-training necessario.")
            return None, None
        model_buy = joblib.load(MODEL_FILE)
        print(f"Modello BUY caricato da {MODEL_FILE} (backward compat, eta': {age_hours:.1f}h)")
    else:
        return None, None

    # Carica SHORT se disponibile
    if os.path.isfile(MODEL_SHORT_FILE):
        age_hours_s = (time.time() - os.path.getmtime(MODEL_SHORT_FILE)) / 3600
        if age_hours_s <= RETRAIN_HOURS:
            model_short = joblib.load(MODEL_SHORT_FILE)
            print(f"Modello SHORT caricato da {MODEL_SHORT_FILE} (eta': {age_hours_s:.1f}h)")
        else:
            print(f"Modello SHORT troppo vecchio ({age_hours_s:.1f}h), SHORT disabilitato.")
    else:
        print("Modello SHORT non trovato, SHORT disabilitato nel bot live.")

    return model_buy, model_short


def retrain_model():
    """
    Scarica dati freschi, rigenera feature, allena nuovi modelli (BUY + SHORT)
    e li salva su disco. Usato per il retraining periodico nel bot live.
    """
    print("\n=== RETRAINING: scarico dati freschi ===")
    df_raw = fetch_ohlcv()
    df_feat = build_features(df_raw)
    print(f"Retraining su {len(df_feat)} righe")
    model_buy = train_model(df_feat)
    model_short = None
    if ENABLE_SHORT:
        model_short = train_model(df_feat, target_label=-1)
    save_model(model_buy, model_short)
    send_telegram(
        f"🔄 <b>Retraining completato</b>\n"
        f"📊 Righe: {len(df_feat)} | BUY + {'SHORT' if model_short else 'LONG-only'}"
    )
    return model_buy, model_short


def _calc_pnl(price, entry_price, qty, position_type):
    """Calcola P&L correttamente per LONG e SHORT."""
    if position_type == "short":
        pnl_usd = (entry_price - price) * qty
        pnl_pct = (entry_price - price) / entry_price
    else:
        pnl_usd = (price - entry_price) * qty
        pnl_pct = (price - entry_price) / entry_price
    return pnl_usd, pnl_pct


def run_bot(model_buy, model_short=None):
    """
    Loop principale del bot:
    - LONG su Binance Spot Demo (BUY/SELL)
    - SHORT su Binance Futures Demo (SELL to open / BUY to close)
    Ogni SLEEP_SECONDS scarica nuovi dati, genera segnali e piazza ordini.

    Include:
    - Stop loss automatico (2% per LONG e SHORT)
    - Filtro trend EMA20 (BUY solo in uptrend, SHORT solo in downtrend)
    - Retraining periodico ogni RETRAIN_HOURS ore
    - Persistenza stato su file JSON (position_type: long/short/None)
    - Notifiche Telegram su trade ed errori
    - Logging trade su CSV
    """
    # === Exchange setup ===
    exchange_spot = get_testnet_exchange("spot")
    exchange_futures = None
    short_enabled = ENABLE_SHORT and model_short is not None
    if short_enabled:
        try:
            exchange_futures = get_testnet_exchange("future")
            print("Futures Demo: connesso (leverage 1x)")
        except Exception as e:
            print(f"[WARN] Futures non disponibile: {e}. SHORT disabilitato.")
            short_enabled = False

    mode_str = "LONG+SHORT" if short_enabled else "LONG-only"
    print(f"\nBot avviato su Binance Demo | {SYMBOL} | {TIMEFRAME} | {mode_str}")
    print("=" * 55)

    # Carica stato precedente (sopravvive a restart)
    entry_price, entry_qty, entry_time, position_type = load_state()
    if entry_price:
        print(f"Stato caricato: {position_type.upper()} @ {entry_price:.2f} "
              f"({entry_qty} BTC)")

    last_retrain = time.time()
    last_status  = time.time()
    STATUS_INTERVAL = 86400
    send_telegram(
        f"🤖 <b>Bot avviato ({mode_str})</b>\n"
        f"📈 {SYMBOL} | {TIMEFRAME}\n"
        f"🔄 Retraining ogni {RETRAIN_HOURS}h\n"
        f"🛡 SL: {STOP_LOSS:.0%} | BUY min: {MIN_PROBA:.0%}"
        + (f" | SHORT min: {SHORT_MIN_PROBA:.0%}" if short_enabled else "")
    )

    consecutive_net_errors = 0
    while True:
        try:
            # --- Retraining periodico ---
            hours_since = (time.time() - last_retrain) / 3600
            if hours_since >= RETRAIN_HOURS:
                print(f"\n[RETRAIN] {hours_since:.1f}h dall'ultimo training")
                try:
                    model_buy, model_short_new = retrain_model()
                    if model_short_new and short_enabled:
                        model_short = model_short_new
                except Exception as e:
                    print(f"[RETRAIN FALLITO] {e}")
                    send_telegram(
                        f"⚠️ <b>Retraining fallito</b>\n"
                        f"<code>{str(e)[:500]}</code>"
                    )
                last_retrain = time.time()

            # --- Fetch dati e segnali ---
            df      = fetch_ohlcv()
            save_price_history(df)
            df_feat = build_features(df)

            last_row   = df_feat[FEATURES].iloc[-1:]
            buy_proba  = model_buy.predict_proba(last_row)[0][1]
            buy_signal = buy_proba >= MIN_PROBA

            short_proba = 0.0
            short_signal = False
            if short_enabled and model_short is not None:
                short_proba = model_short.predict_proba(last_row)[0][1]
                short_signal = short_proba >= SHORT_MIN_PROBA

            # Segnali tecnici di chiusura
            sell_signal  = technical_sell_signal(df_feat).iloc[-1]   # chiudi LONG
            cover_signal = technical_cover_signal(df_feat).iloc[-1]  # chiudi SHORT

            # --- Conflict resolution: se entrambi i modelli sparano -> HOLD ---
            if buy_signal and short_signal:
                buy_signal = False
                short_signal = False
                print(f"  [CONFLICT] BUY ({buy_proba:.0%}) + SHORT ({short_proba:.0%}) -> HOLD")

            # --- Mutual exclusion posizione: non aprire opposta se gia' in posizione ---
            if buy_signal and position_type == "short":
                buy_signal = False  # non aprire LONG mentre SHORT e' aperto
            if short_signal and position_type == "long":
                short_signal = False  # non aprire SHORT mentre LONG e' aperto

            # --- Balance (dall'exchange corretto) ---
            price = df_feat["close"].iloc[-1]
            if position_type == "short" and exchange_futures:
                balance = exchange_futures.fetch_balance()
                usdt = balance.get("USDT", {}).get("free", 0)
                btc = 0.0  # su Futures il BTC non e' nel wallet come su Spot
            else:
                balance = exchange_spot.fetch_balance()
                usdt = balance["USDT"]["free"]
                btc  = balance["BTC"]["free"]

            # --- Log stato ---
            pos_str = f"{position_type.upper()}" if position_type else "FLAT"
            signal_parts = []
            if buy_signal: signal_parts.append(f"BUY({buy_proba:.0%})")
            if short_signal: signal_parts.append(f"SHORT({short_proba:.0%})")
            if sell_signal and position_type == "long": signal_parts.append("CLOSE_LONG")
            if cover_signal and position_type == "short": signal_parts.append("CLOSE_SHORT")
            signal_str = " | ".join(signal_parts) if signal_parts else "HOLD"
            print(
                f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] "
                f"Prezzo: {price:.2f} | {pos_str} | Signal: {signal_str} | "
                f"USDT: {usdt:.2f}"
            )

            # --- Dashboard ---
            try:
                save_dashboard_data(
                    price, buy_proba, signal_str, usdt, btc,
                    entry_price, entry_qty,
                    df_feat[FEATURES].iloc[-1].to_dict()
                )
            except Exception as e:
                print(f"[DASHBOARD] Errore: {e}")

            # --- Notifica stato periodica ---
            if (time.time() - last_status) >= STATUS_INTERVAL:
                try:
                    now_str = pd.Timestamp.now().strftime("%H:%M")
                    if entry_price and position_type:
                        pnl_usd, pnl_pct = _calc_pnl(price, entry_price, entry_qty or 0, position_type)
                        pnl_icon = "📈" if pnl_usd >= 0 else "📉"
                        pos_icon = "🟢" if position_type == "long" else "🔴"
                        send_telegram(
                            f"📊 <b>Status {now_str}</b>\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"{pos_icon} {position_type.upper()}\n"
                            f"📌 Entry: ${entry_price:,.2f}\n"
                            f"💰 Prezzo: ${price:,.2f}\n"
                            f"{pnl_icon} P&amp;L: {pnl_pct:+.2%} (${pnl_usd:+,.2f})\n"
                            f"🎯 BUY: {buy_proba:.0%} | SHORT: {short_proba:.0%}"
                        )
                    else:
                        send_telegram(
                            f"📊 <b>Status {now_str}</b>\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"⚪ Nessuna posizione\n"
                            f"💵 USDT: ${usdt:,.2f}\n"
                            f"💰 BTC: ${price:,.2f}\n"
                            f"🎯 BUY: {buy_proba:.0%} | SHORT: {short_proba:.0%}"
                        )
                except Exception as e:
                    print(f"[STATUS] Errore: {e}")
                last_status = time.time()

            # ============================================
            # STOP LOSS (SEMPRE attivo, prima di tutto)
            # ============================================

            # Stop loss LONG
            if position_type == "long" and entry_price and btc > 0.0001:
                loss_pct = (price - entry_price) / entry_price
                if loss_pct <= -STOP_LOSS:
                    qty = int(btc * 1e6) / 1e6
                    pnl_usd = (price - entry_price) * qty
                    try:
                        exchange_spot.create_order(SYMBOL, "market", "sell", qty)
                    except Exception as e:
                        print(f"  [ERRORE] Stop loss LONG fallito: {e}")
                        send_telegram(f"🚨 <b>Stop Loss LONG fallito</b>\n<code>{e}</code>")
                        continue
                    entry_price = None; entry_qty = None; entry_time = None; position_type = None
                    save_state(entry_price, entry_qty, entry_time, position_type)
                    print(f"  -> STOP LOSS LONG: SELL {qty} BTC @ {price:.2f} (P&L: ${pnl_usd:+,.2f})")
                    log_trade("SELL(SL)", qty, price, "STOP_LOSS_LONG", buy_proba, pnl_usd)
                    send_telegram(
                        f"🛑 <b>Stop Loss LONG</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"🔴 SELL {qty} BTC @ ${price:,.2f}\n"
                        f"📉 Loss: {loss_pct:.2%} | P&amp;L: ${pnl_usd:+,.2f}"
                    )
                    time.sleep(SLEEP_SECONDS)
                    continue  # NON aprire nuove posizioni nello stesso ciclo

            # Stop loss SHORT
            if position_type == "short" and entry_price and exchange_futures:
                loss_pct = (entry_price - price) / entry_price  # prezzo sale = perdita
                if loss_pct <= -SHORT_STOP_LOSS:
                    qty = entry_qty or 0
                    pnl_usd = (entry_price - price) * qty
                    try:
                        exchange_futures.create_order(SYMBOL, "market", "buy", qty)
                    except Exception as e:
                        print(f"  [ERRORE] Stop loss SHORT fallito: {e}")
                        send_telegram(f"🚨 <b>Stop Loss SHORT fallito</b>\n<code>{e}</code>")
                        continue
                    entry_price = None; entry_qty = None; entry_time = None; position_type = None
                    save_state(entry_price, entry_qty, entry_time, position_type)
                    print(f"  -> STOP LOSS SHORT: BUY {qty} BTC @ {price:.2f} (P&L: ${pnl_usd:+,.2f})")
                    log_trade("BUY(SL)", qty, price, "STOP_LOSS_SHORT", short_proba, pnl_usd)
                    send_telegram(
                        f"🛑 <b>Stop Loss SHORT</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"🟢 BUY(cover) {qty} BTC @ ${price:,.2f}\n"
                        f"📈 Loss: {loss_pct:.2%} | P&amp;L: ${pnl_usd:+,.2f}"
                    )
                    time.sleep(SLEEP_SECONDS)
                    continue

            # ============================================
            # HOLD MINIMO (sopprime chiusura tecnica)
            # ============================================
            if sell_signal and position_type == "long" and entry_time:
                hours_held = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
                if hours_held < MIN_HOLD_BARS:
                    sell_signal = False
                    print(f"  -> CLOSE LONG soppresso: hold {hours_held:.1f}h < {MIN_HOLD_BARS}h")

            if cover_signal and position_type == "short" and entry_time:
                hours_held = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
                if hours_held < SHORT_MIN_HOLD:
                    cover_signal = False
                    print(f"  -> CLOSE SHORT soppresso: hold {hours_held:.1f}h < {SHORT_MIN_HOLD}h")

            # ============================================
            # APERTURA POSIZIONI (solo se flat)
            # ============================================

            if buy_signal and usdt > 10 and not position_type:
                trend_ok = df_feat["trend_up"].iloc[-1] == 1
                if not trend_ok:
                    print(f"  -> Trend ribassista (prezzo < EMA20), BUY bloccato.")
                else:
                    qty = round((usdt * TRADE_SIZE) / price, 6)
                    try:
                        exchange_spot.create_order(SYMBOL, "market", "buy", qty)
                    except Exception as e:
                        print(f"  [ERRORE] BUY fallito: {e}")
                        send_telegram(f"🚨 <b>BUY fallito</b>\n<code>{e}</code>")
                        continue
                    entry_price = price; entry_qty = qty
                    entry_time = datetime.now(timezone.utc); position_type = "long"
                    save_state(entry_price, entry_qty, entry_time, position_type)
                    print(f"  -> ORDER: BUY {qty} BTC @ {price:.2f} (prob: {buy_proba:.0%})")
                    log_trade("BUY", qty, price, "BUY", buy_proba)
                    send_telegram(
                        f"🟢 <b>BUY eseguito</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"🛒 {qty} BTC @ ${price:,.2f}\n"
                        f"🤖 Confidenza: {buy_proba:.0%}\n"
                        f"💵 Investito: ${qty * price:,.2f}"
                    )

            elif short_signal and short_enabled and usdt > 10 and not position_type:
                trend_down = df_feat["trend_up"].iloc[-1] == 0
                if not trend_down:
                    print(f"  -> Trend rialzista (prezzo > EMA20), SHORT bloccato.")
                else:
                    qty = round((usdt * SHORT_TRADE_SIZE) / price, 6)
                    try:
                        exchange_futures.create_order(SYMBOL, "market", "sell", qty)
                    except Exception as e:
                        print(f"  [ERRORE] SHORT fallito: {e}")
                        send_telegram(f"🚨 <b>SHORT fallito</b>\n<code>{e}</code>")
                        continue
                    entry_price = price; entry_qty = qty
                    entry_time = datetime.now(timezone.utc); position_type = "short"
                    save_state(entry_price, entry_qty, entry_time, position_type)
                    print(f"  -> ORDER: SHORT {qty} BTC @ {price:.2f} (prob: {short_proba:.0%})")
                    log_trade("SHORT", qty, price, "SHORT", short_proba)
                    send_telegram(
                        f"🔴 <b>SHORT eseguito</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"📉 {qty} BTC @ ${price:,.2f}\n"
                        f"🤖 Confidenza: {short_proba:.0%}\n"
                        f"💵 Margine: ${qty * price:,.2f}"
                    )

            # ============================================
            # CHIUSURA POSIZIONI (segnali tecnici)
            # ============================================

            elif sell_signal and position_type == "long" and btc > 0.0001:
                qty = int(btc * 1e6) / 1e6
                pnl_usd, pnl_pct = _calc_pnl(price, entry_price, qty, "long")
                try:
                    exchange_spot.create_order(SYMBOL, "market", "sell", qty)
                except Exception as e:
                    print(f"  [ERRORE] CLOSE LONG fallito: {e}")
                    send_telegram(f"🚨 <b>CLOSE LONG fallito</b>\n<code>{e}</code>")
                    continue
                entry_price = None; entry_qty = None; entry_time = None; position_type = None
                save_state(entry_price, entry_qty, entry_time, position_type)
                print(f"  -> ORDER: CLOSE LONG {qty} BTC @ {price:.2f} (P&L: ${pnl_usd:+,.2f})")
                log_trade("SELL", qty, price, "SELL_TECH", buy_proba, pnl_usd)
                pnl_icon = "📈" if pnl_usd >= 0 else "📉"
                send_telegram(
                    f"🔴 <b>CLOSE LONG</b>\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"💼 {qty} BTC @ ${price:,.2f}\n"
                    f"{pnl_icon} P&amp;L: {pnl_pct:+.2%} (${pnl_usd:+,.2f})"
                )

            elif cover_signal and position_type == "short" and exchange_futures:
                qty = entry_qty or 0
                pnl_usd, pnl_pct = _calc_pnl(price, entry_price, qty, "short")
                try:
                    exchange_futures.create_order(SYMBOL, "market", "buy", qty)
                except Exception as e:
                    print(f"  [ERRORE] CLOSE SHORT fallito: {e}")
                    send_telegram(f"🚨 <b>CLOSE SHORT fallito</b>\n<code>{e}</code>")
                    continue
                entry_price = None; entry_qty = None; entry_time = None; position_type = None
                save_state(entry_price, entry_qty, entry_time, position_type)
                print(f"  -> ORDER: CLOSE SHORT {qty} BTC @ {price:.2f} (P&L: ${pnl_usd:+,.2f})")
                log_trade("COVER", qty, price, "COVER_TECH", short_proba, pnl_usd)
                pnl_icon = "📈" if pnl_usd >= 0 else "📉"
                send_telegram(
                    f"🟢 <b>CLOSE SHORT</b>\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"💼 BUY(cover) {qty} BTC @ ${price:,.2f}\n"
                    f"{pnl_icon} P&amp;L: {pnl_pct:+.2%} (${pnl_usd:+,.2f})"
                )

            else:
                print(f"  -> HOLD (nessuna azione)")

            consecutive_net_errors = 0

        except TRANSIENT_ERRORS as e:
            consecutive_net_errors += 1
            err_msg = f"{type(e).__name__}: {e}"
            if consecutive_net_errors <= MAX_RETRIES:
                wait = RETRY_BACKOFF[consecutive_net_errors - 1]
                print(f"[RETRY {consecutive_net_errors}/{MAX_RETRIES}] {err_msg}")
                print(f"  -> Riprovo tra {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"[ERRORE] {err_msg} (dopo {MAX_RETRIES} tentativi)")
                send_telegram(
                    f"🚨 <b>Errore di rete</b> (dopo {MAX_RETRIES} tentativi)\n"
                    f"<code>{err_msg[:500]}</code>"
                )
                consecutive_net_errors = 0

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[ERRORE] {err_msg}")
            send_telegram(
                f"🚨 <b>Errore</b>\n"
                f"<code>{err_msg[:500]}</code>"
            )
            consecutive_net_errors = 0

        print(f"  -> Prossimo ciclo tra {SLEEP_SECONDS // 60} minuti...\n")
        time.sleep(SLEEP_SECONDS)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    backtest_only = "--backtest" in sys.argv

    if backtest_only:
        # === Modalita' backtest-only: usa modelli salvati, skip training ===
        print("=== MODALITA' BACKTEST-ONLY ===\n")
        model_buy, model_short = load_model()
        if model_buy is None:
            print("ERRORE: nessun modello salvato. Esegui prima il training completo.")
            sys.exit(1)

        print("=== Fase 1 - download dati ===")
        df_raw = fetch_ohlcv()
        print(f"Scaricate {len(df_raw)} candele per {SYMBOL} ({TIMEFRAME})")

        print("\n=== Fase 2 - feature engineering ===")
        df_feat = build_features(df_raw)
        print(f"Dataset: {len(df_feat)} righe, {len(FEATURES)} features")

        # Walk-forward per generare predizioni out-of-sample (usa params cached)
        best_params_buy = optimize_hyperparams(df_feat, target_label=1, cache_file=OPTUNA_BUY_FILE)
        best_params_short = None
        if ENABLE_SHORT:
            best_params_short = optimize_hyperparams(df_feat, target_label=-1, cache_file=OPTUNA_SHORT_FILE)

        print("\n=== Fase 3 - walk-forward (solo predizioni, params cached) ===")
        model_buy, model_short, wf_predictions = walk_forward_train(
            df_feat, best_params_buy=best_params_buy, best_params_short=best_params_short
        )

        print("\n=== Fase 4 - backtest ===")
        backtest(df_raw, df_feat, model_buy, model_short, pred_series=wf_predictions)
        sys.exit(0)

    # === Modalita' normale: training completo + (opzionale) bot live ===
    model_buy, model_short = load_model()

    if model_buy is not None:
        print("Modello caricato da disco - skip training.")
        print(f"Per forzare il retraining, cancella {MODEL_BUY_FILE} e {MODEL_SHORT_FILE}\n")
        try:
            df_raw = fetch_ohlcv()
            save_price_history(df_raw)
        except Exception:
            pass
    else:
        print("=== CRYPTOBOT: fase 1 - download dati ===")
        df_raw  = fetch_ohlcv()
        save_price_history(df_raw)
        print(f"Scaricate {len(df_raw)} candele per {SYMBOL} ({TIMEFRAME})")

        print("\n=== CRYPTOBOT: fase 2 - feature engineering ===")
        df_feat = build_features(df_raw)
        print(f"Dataset dopo feature engineering: {len(df_feat)} righe, {len(FEATURES)} features")

        # Fase 2b: Ottimizzazione iperparametri (separata per BUY e SHORT)
        print("\n=== CRYPTOBOT: fase 2b - ottimizzazione iperparametri BUY ===")
        best_params_buy = optimize_hyperparams(df_feat, target_label=1, cache_file=OPTUNA_BUY_FILE)

        best_params_short = None
        if ENABLE_SHORT:
            print("\n=== CRYPTOBOT: fase 2b - ottimizzazione iperparametri SHORT ===")
            best_params_short = optimize_hyperparams(df_feat, target_label=-1, cache_file=OPTUNA_SHORT_FILE)

        # Fase 3: Walk-forward (dual model)
        print("\n=== CRYPTOBOT: fase 3 - walk-forward training (LONG+SHORT) ===")
        model_buy, model_short, wf_predictions = walk_forward_train(
            df_feat, best_params_buy=best_params_buy, best_params_short=best_params_short
        )

        # Feature importance BUY (ultimo fold)
        print("\n=== Feature Importance BUY (ultimo fold) ===")
        importances = model_buy.feature_importances_
        feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
        for feat, imp in feat_imp:
            bar = "#" * int(imp * 50)
            print(f"  {feat:>15}: {imp:.4f} {bar}")

        # Feature importance SHORT (se disponibile)
        if model_short is not None:
            print("\n=== Feature Importance SHORT (ultimo fold) ===")
            importances_s = model_short.feature_importances_
            feat_imp_s = sorted(zip(FEATURES, importances_s), key=lambda x: x[1], reverse=True)
            for feat, imp in feat_imp_s:
                bar = "#" * int(imp * 50)
                print(f"  {feat:>15}: {imp:.4f} {bar}")

        # Salva i modelli per riusi futuri
        save_model(model_buy, model_short)

        # Fase 4: Backtest
        print("\n=== CRYPTOBOT: fase 4 - backtest (walk-forward out-of-sample, LONG+SHORT) ===")
        backtest(df_raw, df_feat, model_buy, model_short, pred_series=wf_predictions)

    # Decommenta la riga sotto per avviare il bot live sul testnet
    #run_bot(model_buy, model_short)

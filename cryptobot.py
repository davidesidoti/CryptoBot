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
FETCH_LIMIT     = 5000        # quante candele storiche scaricare (paginato)
N_TRAIN         = 400         # candele usate per il training
FUTURE_BARS     = 3           # quante candele in avanti per calcolare il label
SIGNAL_THRESH   = 0.007       # 0.7% di movimento minimo per generare segnale
MIN_PROBA       = 0.60        # confidenza minima per eseguire un ordine
TRADE_SIZE      = 0.95        # % del capitale usata per ogni trade
STOP_LOSS       = 0.02        # 2% stop loss — chiude posizione se la perdita supera questa soglia
INITIAL_CASH    = 500         # capitale iniziale per il backtest (in USD)
SLEEP_SECONDS   = 1800        # secondi tra ogni ciclo del bot (30 min)
LOG_FILE        = "trades_log.csv"
RETRAIN_HOURS   = 24            # riaddestra il modello ogni N ore
MODEL_FILE      = "model.joblib"
STATE_FILE      = "bot_state.json"
DASHBOARD_FILE  = "dashboard_data.json"

FEATURES = [
    "rsi", "macd", "macd_signal", "bb_width",
    "vol_change", "price_change", "ema_cross",
    "atr", "obv_change", "stoch_k", "rsi_slope", "hour",
    # Multi-timeframe features (resampled da 1h)
    "rsi_4h", "macd_4h", "ema_cross_4h", "trend_4h",
    "rsi_1d", "trend_1d",
    # Regime / volatilita'
    "atr_ratio", "vol_regime"
]

# Walk-forward validation
WF_TRAIN_BARS   = 1500        # candele per ogni finestra di training
WF_TEST_BARS    = 200         # candele per ogni finestra di test

# ─────────────────────────────────────────────
# 1. FETCH DATI
# ─────────────────────────────────────────────

def fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=FETCH_LIMIT):
    """
    Scarica i dati OHLCV da Binance (endpoint pubblico, niente API key).
    Supporta paginazione per ottenere piu' di 1000 candele.
    """
    exchange = ccxt.binance()
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
    ema20_1d          = ta.ema(df_1d["close"], length=20)
    df_1d["trend_1d"] = (df_1d["close"] > ema20_1d).astype(int)
    for col in ["rsi_1d", "trend_1d"]:
        df[col] = df_1d[col].reindex(df.index, method="ffill")

    # === Regime / volatilita' ===
    atr_fast = ta.atr(df["high"], df["low"], df["close"], length=7)
    atr_slow = ta.atr(df["high"], df["low"], df["close"], length=28)
    df["atr_ratio"]  = atr_fast / atr_slow  # >1 = volatilita' in aumento
    vol_20 = df["volume"].rolling(20).mean()
    df["vol_regime"] = df["volume"] / vol_20  # >1 = volume sopra media

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

def train_model(df):
    """
    Allena un XGBoost binary classifier: BUY (1) vs NO-BUY (0).
    Il segnale SELL viene gestito da regole tecniche, non dal classificatore.
    Usa shuffle=False perche' i dati sono time-series.
    """
    X = df[FEATURES]
    # Binary: 1 = BUY, 0 = tutto il resto (HOLD + SELL)
    y = (df["label"] == 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Diagnostica distribuzione classi
    print("Distribuzione classi (train):")
    for cls in [0, 1]:
        count = (y_train == cls).sum()
        name = "BUY" if cls == 1 else "NO-BUY"
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
        early_stopping_rounds=40
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print(f"\nMiglior iterazione: {model.best_iteration} / 500")
    print("\n=== Valutazione modello sul test set ===")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=["NO-BUY", "BUY"]))

    return model


# ─────────────────────────────────────────────
# 3b. WALK-FORWARD VALIDATION
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

    # RSI ipercomprato + inizio discesa
    sell |= (df["rsi"] > 70) & (df["rsi_slope"] < 0)

    # MACD cross ribassista
    sell |= (df["macd"] < df["macd_signal"]) & (df["macd"].diff() < 0)

    # Prezzo crolla sotto EMA20 con momentum
    sell |= (df["trend_up"] == 0) & (df["price_change"] < -0.005)

    return sell


def walk_forward_train(df):
    """
    Walk-forward validation con binary classification (BUY vs NO-BUY).
    Il segnale SELL e' generato da regole tecniche, non dal modello ML.
    Per ogni finestra, genera predictions out-of-sample.
    Ritorna il modello allenato sull'ultimo fold e le predictions aggregate.
    """
    X = df[FEATURES]
    # Binary target: 1 = BUY, 0 = NO-BUY
    y = (df["label"] == 1).astype(int)

    all_preds  = []
    all_proba  = []
    all_actual = []
    all_idx    = []
    n = len(df)
    fold = 0

    print(f"Walk-forward: train={WF_TRAIN_BARS}, test={WF_TEST_BARS}, "
          f"dataset={n} righe")

    i = 0
    while i + WF_TRAIN_BARS + WF_TEST_BARS <= n:
        fold += 1
        train_end = i + WF_TRAIN_BARS
        test_end  = min(train_end + WF_TEST_BARS, n)

        X_train = X.iloc[i:train_end]
        y_train = y.iloc[i:train_end]
        X_test  = X.iloc[train_end:test_end]
        y_test  = y.iloc[train_end:test_end]

        # scale_pos_weight per questo fold
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
            early_stopping_rounds=40
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Guard: se il modello e' degenere (best_iter troppo basso),
        # tratta tutte le predictions come NO-BUY per evitare falsi segnali
        is_degenerate = model.best_iteration < 10
        if is_degenerate:
            preds = np.zeros(len(X_test), dtype=int)
            proba = np.full(len(X_test), 0.0)
        else:
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]  # probabilita' BUY

        all_preds.extend(preds)
        all_proba.extend(proba)
        all_actual.extend(y_test.values)
        all_idx.extend(y_test.index)

        buy_preds = sum(preds)
        status = " [DEGENERATE - skipped]" if is_degenerate else ""
        print(f"  Fold {fold}: train [{i}:{train_end}] | "
              f"test [{train_end}:{test_end}] | "
              f"best_iter={model.best_iteration} | "
              f"BUY pred: {buy_preds}/{len(preds)}{status}")

        i += WF_TEST_BARS  # scorrimento della finestra

    print(f"\n=== Walk-forward: {fold} fold completati ===")
    print(f"Predictions out-of-sample totali: {len(all_preds)}")
    print("\n=== Classification report (aggregato walk-forward) ===")
    print(classification_report(
        all_actual, all_preds,
        target_names=["NO-BUY", "BUY"]
    ))

    # Analisi qualita' segnali BUY
    preds_arr = np.array(all_preds)
    actual_arr = np.array(all_actual)
    if preds_arr.sum() > 0:
        buy_precision = actual_arr[preds_arr == 1].mean()
        print(f"BUY precision effettiva: {buy_precision:.1%} "
              f"({preds_arr.sum()} segnali BUY predetti)")

    # Genera segnali combinati per il backtest:
    # ML predice BUY, regole tecniche generano SELL
    sell_signals = technical_sell_signal(df)

    # Crea la serie di predictions per il backtest
    # +1 = BUY (dal modello ML), -1 = SELL (da regole tecniche), 0 = HOLD
    pred_values = np.zeros(len(all_preds))
    for j in range(len(all_preds)):
        idx = all_idx[j]
        if all_preds[j] == 1 and all_proba[j] >= MIN_PROBA:
            pred_values[j] = 1   # BUY (ML dice BUY con alta confidenza)
        elif sell_signals.loc[idx]:
            pred_values[j] = -1  # SELL (regole tecniche)
        # else: 0 (HOLD)

    pred_series = pd.Series(pred_values.astype(int), index=all_idx)

    return model, pred_series


# ─────────────────────────────────────────────
# 4. BACKTEST
# ─────────────────────────────────────────────

def backtest(df_raw, df_feat, model, pred_series=None, test_size=0.2):
    """
    Backtesta la strategia ML solo sui dati out-of-sample (test set)
    per evitare data leakage. Include stop loss e report P&L in dollari.
    Se pred_series e' fornita (da walk-forward), usa quelle predictions.
    """

    if pred_series is not None:
        # Walk-forward: usa le predictions aggregate out-of-sample
        feat_index  = pred_series.index
        raw_aligned = df_raw.loc[feat_index].copy()
    else:
        # Singolo split: genera predictions dal modello
        split_idx   = int(len(df_feat) * (1 - test_size))
        df_test     = df_feat.iloc[split_idx:]
        feat_index  = df_test.index
        raw_aligned = df_raw.loc[feat_index].copy()
        predictions = model.predict(df_test[FEATURES]) - 1
        pred_series = pd.Series(predictions, index=feat_index)

    # Filtro trend: blocca BUY quando prezzo < EMA20
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

        def next(self):
            sig = self.signal[-1]
            price = self.data.Close[-1]
            trend_ok = self.trend[-1] == 1

            # Stop loss: chiudi long se la perdita supera STOP_LOSS
            if self.position.is_long and self.entry_price:
                loss = (price - self.entry_price) / self.entry_price
                if loss <= -STOP_LOSS:
                    self.position.close()
                    self.entry_price = None
                    return

            # BUY: entra long (solo se trend up e non gia' in posizione)
            if sig == 1 and not self.position and trend_ok:
                self.buy(size=TRADE_SIZE)
                self.entry_price = price

            # SELL: chiudi long (solo se in posizione)
            elif sig == -1 and self.position.is_long:
                self.position.close()
                self.entry_price = None

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

def get_testnet_exchange():
    """
    Ritorna un'istanza ccxt connessa al Binance Demo Trading (ex Testnet).
    Usa enable_demo_trading() che punta a demo-api.binance.com
    (il vecchio testnet.binance.vision non e' piu' attivo).
    """
    exchange = ccxt.binance({
        "apiKey": TESTNET_API_KEY,
        "secret": TESTNET_SECRET,
        "options": {
            "defaultType": "spot",
        },
    })
    exchange.enable_demo_trading(True)
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

def save_state(entry_price, entry_qty):
    """Salva lo stato della posizione su file JSON."""
    with open(STATE_FILE, "w") as f:
        json.dump({"entry_price": entry_price, "entry_qty": entry_qty}, f)


def load_state():
    """Carica lo stato della posizione da file JSON."""
    if os.path.isfile(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
            return s.get("entry_price"), s.get("entry_qty")
        except (json.JSONDecodeError, KeyError):
            pass
    return None, None


def save_dashboard_data(price, buy_proba, signal_str, usdt, btc,
                        entry_price, entry_qty, features_row):
    """Salva snapshot del ciclo corrente per la dashboard web."""
    data = {
        "timestamp": pd.Timestamp.utcnow().isoformat() + "Z",
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
        "sleep_seconds": SLEEP_SECONDS,
        "stop_loss": STOP_LOSS,
        "min_proba": MIN_PROBA,
    }
    with open(DASHBOARD_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_model(model):
    """Salva il modello su disco con joblib."""
    joblib.dump(model, MODEL_FILE)
    print(f"Modello salvato in {MODEL_FILE}")


def load_model():
    """
    Carica il modello da disco se esiste e ha meno di RETRAIN_HOURS ore.
    Ritorna il modello o None.
    """
    if not os.path.isfile(MODEL_FILE):
        return None
    age_hours = (time.time() - os.path.getmtime(MODEL_FILE)) / 3600
    if age_hours > RETRAIN_HOURS:
        print(f"Modello trovato ma troppo vecchio ({age_hours:.1f}h). Ri-training necessario.")
        return None
    model = joblib.load(MODEL_FILE)
    print(f"Modello caricato da {MODEL_FILE} (eta': {age_hours:.1f}h)")
    return model


def retrain_model():
    """
    Scarica dati freschi, rigenera feature, allena un nuovo modello
    e lo salva su disco. Usato per il retraining periodico nel bot live.
    """
    print("\n=== RETRAINING: scarico dati freschi ===")
    df_raw = fetch_ohlcv()
    df_feat = build_features(df_raw)
    print(f"Retraining su {len(df_feat)} righe")
    model = train_model(df_feat)
    save_model(model)
    send_telegram(
        f"🔄 <b>Retraining completato</b>\n"
        f"📊 Righe: {len(df_feat)} | Modello aggiornato"
    )
    return model


def run_bot(model):
    """
    Loop principale del bot (solo LONG su Binance Spot):
    ogni SLEEP_SECONDS scarica nuovi dati, genera un segnale e
    piazza ordini sul Binance Testnet.

    Include:
    - Stop loss automatico (2%)
    - Filtro trend EMA20
    - Retraining periodico ogni RETRAIN_HOURS ore
    - Persistenza stato su file JSON
    - Notifiche Telegram su trade ed errori
    - Logging trade su CSV
    """
    exchange = get_testnet_exchange()
    print(f"\nBot avviato su Binance Testnet | {SYMBOL} | {TIMEFRAME}")
    print("=" * 55)

    # Carica stato precedente (sopravvive a restart)
    entry_price, entry_qty = load_state()
    if entry_price:
        print(f"Stato caricato: posizione aperta @ {entry_price:.2f} "
              f"({entry_qty} BTC)")

    last_retrain = time.time()
    last_status  = time.time()
    STATUS_INTERVAL = 86400  # notifica stato ogni 24 ore
    send_telegram(
        f"🤖 <b>Bot avviato</b>\n"
        f"📈 {SYMBOL} | {TIMEFRAME}\n"
        f"🔄 Retraining ogni {RETRAIN_HOURS}h\n"
        f"🛡 Stop loss: {STOP_LOSS:.0%} | Min confidenza: {MIN_PROBA:.0%}"
    )

    while True:
        try:
            # --- Retraining periodico ---
            hours_since = (time.time() - last_retrain) / 3600
            if hours_since >= RETRAIN_HOURS:
                print(f"\n[RETRAIN] {hours_since:.1f}h dall'ultimo training")
                try:
                    model = retrain_model()
                except Exception as e:
                    print(f"[RETRAIN FALLITO] {e}")
                    send_telegram(
                        f"⚠️ <b>Retraining fallito</b>\n"
                        f"<code>{str(e)[:500]}</code>"
                    )
                last_retrain = time.time()  # reset timer anche se fallito

            df      = fetch_ohlcv()
            df_feat = build_features(df)

            last_row   = df_feat[FEATURES].iloc[-1:]
            buy_proba  = model.predict_proba(last_row)[0][1]  # probabilita' BUY
            buy_signal = buy_proba >= MIN_PROBA

            # Segnale SELL tecnico (non ML)
            sell_signal = technical_sell_signal(df_feat).iloc[-1]

            balance = exchange.fetch_balance()
            usdt    = balance["USDT"]["free"]
            btc     = balance["BTC"]["free"]
            price   = df_feat["close"].iloc[-1]

            signal_str = "BUY" if buy_signal else ("SELL(tech)" if sell_signal else "HOLD")
            print(
                f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] "
                f"Prezzo: {price:.2f} | "
                f"Signal: {signal_str} (BUY prob: {buy_proba:.0%}) | "
                f"USDT: {usdt:.2f} | BTC: {btc:.6f}"
            )

            # --- Aggiorna dati dashboard ---
            try:
                save_dashboard_data(
                    price, buy_proba, signal_str, usdt, btc,
                    entry_price, entry_qty,
                    df_feat[FEATURES].iloc[-1].to_dict()
                )
            except Exception as e:
                print(f"[DASHBOARD] Errore salvataggio: {e}")

            # --- Notifica stato periodica (ogni 30 min, solo lettura) ---
            if (time.time() - last_status) >= STATUS_INTERVAL:
                try:
                    now_str = pd.Timestamp.now().strftime("%H:%M")
                    if entry_price and btc > 0.0001:
                        pnl_pct = (price - entry_price) / entry_price
                        pnl_usd = (price - entry_price) * (entry_qty or 0)
                        pnl_icon = "📈" if pnl_usd >= 0 else "📉"
                        send_telegram(
                            f"📊 <b>Status {now_str}</b>\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"🟢 In posizione\n"
                            f"📌 Entry: ${entry_price:,.2f}\n"
                            f"💰 Prezzo: ${price:,.2f}\n"
                            f"{pnl_icon} P&amp;L: {pnl_pct:+.2%} (${pnl_usd:+,.2f})\n"
                            f"🎯 Signal: {signal_str} ({buy_proba:.0%})"
                        )
                    else:
                        send_telegram(
                            f"📊 <b>Status {now_str}</b>\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"⚪ Nessuna posizione\n"
                            f"💵 USDT: ${usdt:,.2f}\n"
                            f"💰 BTC: ${price:,.2f}\n"
                            f"🎯 Signal: {signal_str} ({buy_proba:.0%})"
                        )
                except Exception as e:
                    print(f"[STATUS] Errore notifica: {e}")
                last_status = time.time()

            # Stop loss check (solo per posizioni long)
            if entry_price and btc > 0.0001:
                loss = (price - entry_price) / entry_price
                if loss <= -STOP_LOSS:
                    qty = round(btc * TRADE_SIZE, 6)
                    pnl_usd = (price - entry_price) * qty
                    exchange.create_order(SYMBOL, "market", "sell", qty)
                    print(f"  -> STOP LOSS: SELL {qty} BTC @ {price:.2f} "
                          f"(loss: {loss:.2%}, P&L: ${pnl_usd:+,.2f})")
                    log_trade("SELL(SL)", qty, price, "STOP_LOSS",
                              buy_proba, pnl_usd)
                    send_telegram(
                        f"🛑 <b>Stop Loss</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"🔴 SELL {qty} BTC\n"
                        f"💰 Prezzo: ${price:,.2f}\n"
                        f"📉 Loss: {loss:.2%}\n"
                        f"💸 P&amp;L: ${pnl_usd:+,.2f}"
                    )
                    entry_price = None
                    entry_qty = None
                    save_state(entry_price, entry_qty)
                    print(f"  -> Prossimo ciclo tra {SLEEP_SECONDS // 60} minuti...\n")
                    time.sleep(SLEEP_SECONDS)
                    continue

            if buy_signal and usdt > 10 and not entry_price:
                # Filtro trend: compra solo se prezzo > EMA20
                trend_ok = df_feat["trend_up"].iloc[-1] == 1
                if not trend_ok:
                    print(f"  -> Trend ribassista (prezzo < EMA20), BUY bloccato.")
                else:
                    # BUY: entra long
                    qty = round((usdt * TRADE_SIZE) / price, 6)
                    exchange.create_order(SYMBOL, "market", "buy", qty)
                    print(f"  -> ORDER: BUY {qty} BTC @ {price:.2f} "
                          f"(BUY prob: {buy_proba:.0%})")
                    log_trade("BUY", qty, price, "BUY", buy_proba)
                    send_telegram(
                        f"🟢 <b>BUY eseguito</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"🛒 {qty} BTC @ ${price:,.2f}\n"
                        f"🤖 Confidenza ML: {buy_proba:.0%}\n"
                        f"💵 Investito: ${qty * price:,.2f}"
                    )
                    entry_price = price
                    entry_qty = qty
                    save_state(entry_price, entry_qty)

            elif sell_signal and btc > 0.0001 and entry_price:
                # SELL tecnico: chiudi long
                qty = round(btc * TRADE_SIZE, 6)
                pnl_usd = (price - entry_price) * qty
                pnl_pct = (price - entry_price) / entry_price
                exchange.create_order(SYMBOL, "market", "sell", qty)
                print(f"  -> ORDER: SELL(tech) {qty} BTC @ {price:.2f} "
                      f"(P&L: ${pnl_usd:+,.2f})")
                log_trade("SELL", qty, price, "SELL_TECH", buy_proba, pnl_usd)
                pnl_icon = "📈" if pnl_usd >= 0 else "📉"
                send_telegram(
                    f"🔴 <b>SELL eseguito</b>\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"💼 {qty} BTC @ ${price:,.2f}\n"
                    f"{pnl_icon} P&amp;L: {pnl_pct:+.2%} (${pnl_usd:+,.2f})"
                )
                entry_price = None
                entry_qty = None
                save_state(entry_price, entry_qty)

            else:
                print(f"  -> HOLD (nessuna azione)")

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[ERRORE] {err_msg}")
            send_telegram(
                f"🚨 <b>Errore</b>\n"
                f"<code>{err_msg[:500]}</code>"
            )

        print(f"  -> Prossimo ciclo tra {SLEEP_SECONDS // 60} minuti...\n")
        time.sleep(SLEEP_SECONDS)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Prova a caricare un modello recente da disco
    model = load_model()

    if model is not None:
        print("Modello caricato da disco - skip training.")
        print("Per forzare il retraining, cancella model.joblib\n")
    else:
        print("=== CRYPTOBOT: fase 1 - download dati ===")
        df_raw  = fetch_ohlcv()
        print(f"Scaricate {len(df_raw)} candele per {SYMBOL} ({TIMEFRAME})")

        print("\n=== CRYPTOBOT: fase 2 - feature engineering ===")
        df_feat = build_features(df_raw)
        print(f"Dataset dopo feature engineering: {len(df_feat)} righe")

        print("\n=== CRYPTOBOT: fase 3 - walk-forward training ===")
        model, wf_predictions = walk_forward_train(df_feat)

        # Feature importance dall'ultimo modello
        print("\n=== Feature Importance (ultimo fold) ===")
        importances = model.feature_importances_
        feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
        for feat, imp in feat_imp:
            bar = "#" * int(imp * 50)
            print(f"  {feat:>15}: {imp:.4f} {bar}")

        # Salva il modello per riusi futuri
        save_model(model)

        print("\n=== CRYPTOBOT: fase 4 - backtest (walk-forward out-of-sample) ===")
        backtest(df_raw, df_feat, model, pred_series=wf_predictions)

    # Decommenta la riga sotto per avviare il bot live sul testnet
    run_bot(model)

"""
cryptobot.py
ML-powered crypto trading bot for Binance Testnet.
Fetches OHLCV data, engineers features, trains an XGBoost classifier,
backtests on historical data, and runs live paper trading on Binance Testnet.
"""

import os
import time
import warnings
import ccxt
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

if not TESTNET_API_KEY or not TESTNET_SECRET:
    print("[WARN] BINANCE_TESTNET_API_KEY o BINANCE_TESTNET_SECRET non trovati nel .env.")
    print("       Il bot funzionerà in modalità backtest, ma non potrà piazzare ordini live.")

SYMBOL          = "BTC/USDT"
TIMEFRAME       = "1h"        # candele orarie
FETCH_LIMIT     = 500         # quante candele storiche scaricare
N_TRAIN         = 400         # candele usate per il training
FUTURE_BARS     = 3           # quante candele in avanti per calcolare il label
SIGNAL_THRESH   = 0.005       # 0.5% di movimento minimo per generare segnale
MIN_PROBA       = 0.50        # confidenza minima per eseguire un ordine
TRADE_SIZE      = 0.95        # % del capitale usata per ogni trade
SLEEP_SECONDS   = 3600        # secondo tra ogni ciclo del bot (1h)

FEATURES = [
    "rsi", "macd", "macd_signal", "bb_width",
    "vol_change", "price_change", "ema_cross"
]

# ─────────────────────────────────────────────
# 1. FETCH DATI
# ─────────────────────────────────────────────

def fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=FETCH_LIMIT):
    """
    Scarica i dati OHLCV da Binance (endpoint pubblico, niente API key).
    """
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
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
      +1  = BUY  (prezzo sale > SIGNAL_THRESH nelle prossime FUTURE_BARS candele)
      -1  = SELL (prezzo scende)
       0  = HOLD
    """
    df = df.copy()

    df["rsi"]         = ta.rsi(df["close"], length=14)
    macd_df           = ta.macd(df["close"])
    df["macd"]        = macd_df["MACD_12_26_9"]
    df["macd_signal"] = macd_df["MACDs_12_26_9"]
    bb                = ta.bbands(df["close"], length=20)
    df["bb_upper"]    = bb["BBU_20_2.0"]
    df["bb_lower"]    = bb["BBL_20_2.0"]
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    df["vol_change"]  = df["volume"].pct_change()
    df["price_change"]= df["close"].pct_change()
    df["ema_9"]       = ta.ema(df["close"], length=9)
    df["ema_21"]      = ta.ema(df["close"], length=21)
    df["ema_cross"]   = df["ema_9"] - df["ema_21"]

    future_return     = df["close"].shift(-FUTURE_BARS) / df["close"] - 1
    df["label"]       = 0
    df.loc[future_return >  SIGNAL_THRESH, "label"] =  1
    df.loc[future_return < -SIGNAL_THRESH, "label"] = -1

    return df.dropna()


# ─────────────────────────────────────────────
# 3. TRAINING MODELLO
# ─────────────────────────────────────────────

def train_model(df):
    """
    Allena un XGBoost classifier sul dataset storico.
    Usa shuffle=False perche' i dati sono time-series.
    """
    X = df[FEATURES]
    y = df["label"] + 1   # XGBoost vuole classi 0, 1, 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="mlogloss",
        verbosity=0
    )
    model.fit(X_train, y_train)

    print("\n=== Valutazione modello sul test set ===")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=["SELL", "HOLD", "BUY"]))

    return model


# ─────────────────────────────────────────────
# 4. BACKTEST
# ─────────────────────────────────────────────

def backtest(df_raw, df_feat, model):
    """
    Backtesta la strategia ML su dati storici con backtesting.py.
    """

    # Pre-calcola le predizioni sull'intero dataset e allineale all'indice raw
    feat_index  = df_feat.index
    raw_aligned = df_raw.loc[feat_index].copy()

    predictions = model.predict(df_feat[FEATURES]) - 1   # ritorna a -1, 0, 1
    pred_series = pd.Series(predictions, index=feat_index)

    class MLStrategy(Strategy):
        def init(self):
            self.signal = self.I(
                lambda: pred_series.reindex(raw_aligned.index).fillna(0).values,
                name="ML Signal"
            )

        def next(self):
            sig = self.signal[-1]

            if sig == 1 and not self.position.is_long:
                self.position.close()
                self.buy(size=TRADE_SIZE)

            elif sig == -1 and not self.position.is_short:
                self.position.close()
                self.sell(size=TRADE_SIZE)

    # backtesting.py richiede colonne con la prima lettera maiuscola
    bt_df = raw_aligned.rename(columns={
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume"
    })

    bt = Backtest(
        bt_df,
        MLStrategy,
        cash=10_000,
        commission=0.001,      # 0.1% commission, uguale a Binance
        exclusive_orders=True
    )
    stats = bt.run()

    print("\n=== Risultati Backtest ===")
    print(stats)
    bt.plot()

    return stats


# ─────────────────────────────────────────────
# 5. PAPER TRADING LIVE (Binance Testnet)
# ─────────────────────────────────────────────

def get_testnet_exchange():
    """
    Ritorna un'istanza ccxt connessa al Binance Testnet.
    """
    exchange = ccxt.binance({
        "apiKey": TESTNET_API_KEY,
        "secret": TESTNET_SECRET,
        "options": {
            "defaultType": "spot",
            "hostname": "testnet.binance.vision"
        },
        "urls": {
            "api": {
                "public":  "https://testnet.binance.vision/api",
                "private": "https://testnet.binance.vision/api",
            }
        }
    })
    return exchange


def run_bot(model):
    """
    Loop principale del bot:
    ogni SLEEP_SECONDS scarica nuovi dati, genera un segnale e
    piazza ordini sul Binance Testnet.
    """
    exchange = get_testnet_exchange()
    print(f"\nBot avviato su Binance Testnet | {SYMBOL} | {TIMEFRAME}")
    print("=" * 55)

    while True:
        try:
            df      = fetch_ohlcv()
            df_feat = build_features(df)

            last_row   = df_feat[FEATURES].iloc[-1:]
            proba      = model.predict_proba(last_row)[0]
            pred_class = model.predict(last_row)[0]
            pred_signal = pred_class - 1           # da 0/1/2 a -1/0/1
            confidence  = proba[pred_class]

            balance = exchange.fetch_balance()
            usdt    = balance["USDT"]["free"]
            btc     = balance["BTC"]["free"]
            price   = df_feat["close"].iloc[-1]

            label_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
            print(
                f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] "
                f"Prezzo: {price:.2f} | "
                f"Signal: {label_map[pred_signal]} ({confidence:.0%}) | "
                f"USDT: {usdt:.2f} | BTC: {btc:.6f}"
            )

            # Esegui solo se la confidenza supera la soglia
            if confidence < MIN_PROBA:
                print(f"  -> Confidenza troppo bassa ({confidence:.0%}), skip.")

            elif pred_signal == 1 and usdt > 10:
                qty = round((usdt * TRADE_SIZE) / price, 6)
                exchange.create_order(SYMBOL, "market", "buy", qty)
                print(f"  -> ORDER: BUY {qty} BTC @ {price:.2f}")

            elif pred_signal == -1 and btc > 0.0001:
                qty = round(btc * TRADE_SIZE, 6)
                exchange.create_order(SYMBOL, "market", "sell", qty)
                print(f"  -> ORDER: SELL {qty} BTC @ {price:.2f}")

        except Exception as e:
            print(f"[ERRORE] {type(e).__name__}: {e}")

        print(f"  -> Prossimo ciclo tra {SLEEP_SECONDS // 60} minuti...\n")
        time.sleep(SLEEP_SECONDS)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== CRYPTOBOT: fase 1 - download dati ===")
    df_raw  = fetch_ohlcv()
    print(f"Scaricate {len(df_raw)} candele per {SYMBOL} ({TIMEFRAME})")

    print("\n=== CRYPTOBOT: fase 2 - feature engineering ===")
    df_feat = build_features(df_raw)
    print(f"Dataset dopo feature engineering: {len(df_feat)} righe")

    print("\n=== CRYPTOBOT: fase 3 - training modello ===")
    model = train_model(df_feat)

    print("\n=== CRYPTOBOT: fase 4 - backtest ===")
    backtest(df_raw, df_feat, model)

    # Decommenta la riga sotto per avviare il bot live sul testnet
    # run_bot(model)

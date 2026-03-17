"""
CryptoBot Dashboard — Flask web app per monitorare il bot in tempo reale.

Avvio:
    python dashboard.py

Apri http://localhost:5050 nel browser.
Richiede che il bot (cryptobot.py) sia in esecuzione per aggiornare i dati.
"""

from flask import Flask, render_template, jsonify
import json, csv, os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_FILE = os.path.join(BASE_DIR, "dashboard_data.json")
TRADES_FILE = os.path.join(BASE_DIR, "trades_log.csv")
PRICE_HISTORY_FILE = os.path.join(BASE_DIR, "price_history.json")


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def status():
    """Ritorna lo snapshot corrente del bot."""
    if not os.path.isfile(DASHBOARD_FILE):
        return jsonify({"error": "Bot non ancora avviato"}), 404
    with open(DASHBOARD_FILE) as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/trades")
def trades():
    """Ritorna gli ultimi 50 trade dal CSV."""
    if not os.path.isfile(TRADES_FILE):
        return jsonify([])
    rows = []
    with open(TRADES_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # Ultimi 50, dal piu' recente
    return jsonify(list(reversed(rows[-50:])))


@app.route("/api/equity")
def equity():
    """Calcola equity curve dai trade chiusi."""
    if not os.path.isfile(TRADES_FILE):
        return jsonify([])
    rows = []
    with open(TRADES_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Costruisci equity curve dai SELL (hanno pnl_usd)
    equity = 500.0  # INITIAL_CASH
    curve = [{"timestamp": "", "equity": equity}]
    for row in rows:
        pnl = row.get("pnl_usd", "")
        if pnl and row.get("side", "").startswith("SELL"):
            equity += float(pnl)
            curve.append({
                "timestamp": row["timestamp"],
                "equity": round(equity, 2)
            })
    return jsonify(curve)


@app.route("/api/candles")
def candles():
    """Ritorna le ultime 100 candele OHLCV per il chart candlestick."""
    if not os.path.isfile(PRICE_HISTORY_FILE):
        return jsonify([])
    with open(PRICE_HISTORY_FILE) as f:
        data = json.load(f)
    return jsonify(data)


if __name__ == "__main__":
    print("Dashboard disponibile su http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import asyncio
import websockets
from datetime import datetime

# Configuration de la page
st.set_page_config(page_title="ForexScope Pro", page_icon="🎯", layout="wide")

# TITRE DE L'APPLICATION
st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <h1 style='color: #00D4B2;'>🎯 FOREXSCOPE PRO</h1>
        <p style='color: #888;'>Moteur d'Analyse Non-Repainting & Strategie Pullback/Rejet</p>
    </div>
""", unsafe_allow_html=True)

# SIDEBAR PARAMÈTRES
st.sidebar.header("Configuration")
symbol_choice = st.sidebar.selectbox(
    "Paire d'actifs",
    [
        ("AUD/USD", "frxAUDUSD"),
        ("EUR/USD", "frxEURUSD"),
        ("GBP/USD", "frxGBPUSD"),
        ("USD/JPY", "frxUSDJPY"),
        ("USD/CAD", "frxUSDCAD"),
        ("EUR/GBP", "frxEURGBP"),
    ],
    format_func=lambda x: x[0]
)

timeframe_choice = st.sidebar.selectbox(
    "Unite de temps",
    [
        ("M5 (5 Minutes)", 300),
        ("M15 (15 Minutes)", 900),
        ("M30 (30 Minutes)", 1800),
        ("H1 (1 Heure)", 3600)
    ],
    index=1,
    format_func=lambda x: x[0]
)

# FONCTION WEBSOCKET DERIV
async def fetch_deriv_candles(symbol: str, granularity: int, count: int = 250):
    uri = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    try:
        async with websockets.connect(uri, timeout=10) as ws:
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "style": "candles",
                "granularity": granularity
            }
            await ws.send(json.dumps(request))
            response = await ws.recv()
            data = json.loads(response)

            if "candles" in data:
                df = pd.DataFrame(data["candles"])
                df['time'] = pd.to_datetime(df['epoch'], unit='s')
                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
                df = df[['time', 'Open', 'High', 'Low', 'Close']]
                return df
            else:
                return None
    except Exception as e:
        st.error(f"Erreur Deriv : {e}")
        return None

# MOTEUR DE STRATÉGIE
def analyze_market(df: pd.DataFrame):
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # Bougie clôturée (Index -2) pour éviter tout repeint
    closed = df.iloc[-2]
    prev_closed = df.iloc[-3]
    atr = closed['ATR']

    body = abs(closed['Close'] - closed['Open'])
    upper_wick = closed['High'] - max(closed['Close'], closed['Open'])
    lower_wick = min(closed['Close'], closed['Open']) - closed['Low']

    trend_up = closed['EMA20'] > closed['EMA50'] and closed['Close'] > closed['EMA50']
    trend_down = closed['EMA20'] < closed['EMA50'] and closed['Close'] < closed['EMA50']

    in_buy_zone = closed['Low'] <= closed['EMA20'] or closed['Low'] <= closed['EMA50']
    in_sell_zone = closed['High'] >= closed['EMA20'] or closed['High'] >= closed['EMA50']

    bullish_rejection = lower_wick >= (1.3 * body) or (closed['Close'] > prev_closed['High'])
    bearish_rejection = upper_wick >= (1.3 * body) or (closed['Close'] < prev_closed['Low'])

    verdict = "ATTENDRE"
    color = "#FFA500"
    action = "HOLD"
    motif = "Aucune configuration valide sur la derniere bougie cloturee."
    entry = None
    sl = None
    tp = None

    if trend_up and in_buy_zone and bullish_rejection and (40 <= closed['RSI'] <= 68):
        verdict = "ACHAT (BUY)"
        color = "#00FF88"
        action = "BUY"
        motif = "Tendance haussiere + Pullback EMA + Rejet haussier valide."
        entry = df.iloc[-1]['Open']
        sl = round(closed['Low'] - (0.5 * atr), 5)
        risk = entry - sl
        tp = round(entry + (2 * risk), 5)

    elif trend_down and in_sell_zone and bearish_rejection and (32 <= closed['RSI'] <= 60):
        verdict = "VENTE (SELL)"
        color = "#FF3366"
        action = "SELL"
        motif = "Tendance baissiere + Pullback EMA + Rejet baissier valide."
        entry = df.iloc[-1]['Open']
        sl = round(closed['High'] + (0.5 * atr), 5)
        risk = sl - entry
        tp = round(entry - (2 * risk), 5)

    elif not trend_up and not trend_down:
        motif = "Marche en range / indecision. Patience recommandee."

    return {
        "verdict": verdict,
        "color": color,
        "action": action,
        "motif": motif,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rsi": round(closed['RSI'], 2),
        "atr": round(atr, 5),
        "closed_time": closed['time']
    }

# INTERFACE UTILISATEUR
symbol_name, symbol_code = symbol_choice
tf_name, tf_seconds = timeframe_choice

col_btn, col_blank = st.columns([1, 4])
with col_btn:
    st.button("Actualiser l'analyse", use_container_width=True)

with st.spinner("Connexion a Deriv et calcul en direct..."):
    df_candles = asyncio.run(fetch_deriv_candles(symbol_code, tf_seconds))

if df_candles is not None and len(df_candles) > 60:
    res = analyze_market(df_candles)
    current_price = df_candles.iloc[-1]['Close']

    st.markdown(f"""
        <div style='background-color: #1E222D; border-left: 8px solid {res["color"]}; padding: 18px; border-radius: 8px; margin-bottom: 20px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h4 style='color: #888; margin:0;'>VERDICT {symbol_name} ({tf_name})</h4>
                    <h1 style='color: {res["color"]}; margin: 5px 0 0 0;'>{res["verdict"]}</h1>
                    <p style='color: #DDD; margin: 8px 0 0 0;'><b>Analyse :</b> {res["motif"]}</p>
                </div>
                <div style='text-align: right;'>
                    <span style='color: #888; font-size: 14px;'>Prix Direct</span>
                    <h2 style='color: #FFF; margin: 0;'>{current_price:.5f}</h2>
                    <small style='color: #777;'>Bougie cloturee a {res["closed_time"].strftime("%H:%M UTC")}</small>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if res["action"] in ["BUY", "SELL"]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entree Recommandee", f"{res['entry']:.5f}")
        c2.metric("Stop Loss", f"{res['sl']:.5f}")
        c3.metric("Take Profit", f"{res['tp']:.5f}")
        c4.metric("Ratio R:R", "1 : 2.0")
    else:
        st.info("Statut : En attente d'une configuration a haute probabilite.")

    # Graphique
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_candles['time'],
        open=df_candles['Open'],
        high=df_candles['High'],
        low=df_candles['Low'],
        close=df_candles['Close'],
        name="Prix"
    ))
    fig.add_trace(go.Scatter(x=df_candles['time'], y=df_candles['EMA20'], line=dict(color='#00D4B2', width=1.5), name="EMA 20"))
    fig.add_trace(go.Scatter(x=df_candles['time'], y=df_candles['EMA50'], line=dict(color='#FF9900', width=1.5), name="EMA 50"))

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=450,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    c_rsi, c_atr = st.columns(2)
    c_rsi.metric("RSI (14)", f"{res['rsi']}")
    c_atr.metric("ATR (14)", f"{res['atr']:.5f}")
else:
    st.warning("Impossible de recuperer les donnees. Verifiez votre connexion ou reessayez.")

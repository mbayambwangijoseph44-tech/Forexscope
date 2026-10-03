import streamlit as st
‎import pandas as pd
‎import numpy as np
‎import plotly.graph_objects as go
‎import json
‎import asyncio
‎import websockets
‎from datetime import datetime
‎
‎Configuration de la page
‎st.set_page_config(page_title="ForexScope Pro", page_icon="🎯", layout="wide")
‎
‎--- TITRE DE L'APPLICATION ---
‎st.markdown("""
‎    <div style='text-align: center; padding: 10px;'>
‎        <h1 style='color: #00D4B2;'>🎯 FOREXSCOPE PRO</h1>
‎        <p style='color: #888;'>Moteur d'Analyse Non-Repainting & Stratégie Pullback/Rejet</p>
‎    </div>
‎""", unsafe_allow_html=True)
‎
‎--- SIDEBAR (PARAMÈTRES) ---
‎st.sidebar.header("⚙️ Configuration")
‎symbol_choice = st.sidebar.selectbox(
‎    "Paire d'actifs",
‎    [
‎        ("AUD/USD", "frxAUDUSD"),
‎        ("EUR/USD", "frxEURUSD"),
‎        ("GBP/USD", "frxGBPUSD"),
‎        ("USD/JPY", "frxUSDJPY"),
‎        ("USD/CAD", "frxUSDCAD"),
‎        ("EUR/GBP", "frxEURGBP"),
‎    ],
‎    format_func=lambda x: x[0]
‎)
‎
‎timeframe_choice = st.sidebar.selectbox(
‎    "Unité de temps",
‎    [
‎        ("M5 (5 Minutes)", 300),
‎        ("M15 (15 Minutes)", 900),
‎        ("M30 (30 Minutes)", 1800),
‎        ("H1 (1 Heure)", 3600)
‎    ],
‎    index=1, # M15 par défaut
‎    format_func=lambda x: x[0]
‎)
‎
‎--- FONCTION WEBSOCKET DERIV ---
‎async def fetch_deriv_candles(symbol: str, granularity: int, count: int = 250):
‎    uri = "wss://ws.derivws.com/websockets/v3?app_id=1089"
‎    try:
‎        async with websockets.connect(uri, timeout=10) as ws:
‎            request = {
‎                "ticks_history": symbol,
‎                "adjust_start_time": 1,
‎                "count": count,
‎                "end": "latest",
‎                "style": "candles",
‎                "granularity": granularity
‎            }
‎            await ws.send(json.dumps(request))
‎            response = await ws.recv()
‎            data = json.loads(response)
‎
‎            if "candles" in data:
‎                df = pd.DataFrame(data["candles"])
‎                df['time'] = pd.to_datetime(df['epoch'], unit='s')
‎                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
‎                df = df[['time', 'Open', 'High', 'Low', 'Close']]
‎                return df
‎            else:
‎                return None
‎    except Exception as e:
‎        st.error(f"Erreur de connexion Deriv : {e}")
‎        return None
‎
‎--- MOTEUR DE STRATÉGIE (PULLBACK & REJET STRICT) ---
‎def analyze_market(df: pd.DataFrame):
‎1. Calcul des indicateurs
‎    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
‎    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
‎
‎RSI (14)
‎    delta = df['Close'].diff()
‎    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
‎    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
‎    rs = gain / (loss + 1e-9)
‎    df['RSI'] = 100 - (100 / (1 + rs))
‎
‎ATR (14)
‎    df['TR'] = np.maximum(
‎        df['High'] - df['Low'],
‎        np.maximum(
‎            abs(df['High'] - df['Close'].shift()),
‎            abs(df['Low'] - df['Close'].shift())
‎        )
‎    )
‎    df['ATR'] = df['TR'].rolling(window=14).mean()
‎
‎2. On analyse UNIQUEMENT la dernière bougie CLÔTURÉE (index -2)
‎L'index -1 est la bougie en cours de formation (on ne la prend pas pour éviter le repainting)
‎    closed = df.iloc[-2]
‎    prev_closed = df.iloc[-3]
‎    atr = closed['ATR']
‎
‎    body = abs(closed['Close'] - closed['Open'])
‎    upper_wick = closed['High'] - max(closed['Close'], closed['Open'])
‎    lower_wick = min(closed['Close'], closed['Open']) - closed['Low']
‎    total_range = closed['High'] - closed['Low']
‎
‎Détection de tendance
‎    trend_up = closed['EMA20'] > closed['EMA50'] and closed['Close'] > closed['EMA50']
‎    trend_down = closed['EMA20'] < closed['EMA50'] and closed['Close'] < closed['EMA50']
‎
‎Détection de Pullback dans la zone de valeur
‎    in_buy_zone = closed['Low'] <= closed['EMA20'] or closed['Low'] <= closed['EMA50']
‎    in_sell_zone = closed['High'] >= closed['EMA20'] or closed['High'] >= closed['EMA50']
‎
‎Détection de Rejet (Pinbar ou fort momentum de clôture)
‎    bullish_rejection = lower_wick >= (1.5 * body) or (closed['Close'] > prev_closed['High'])
‎    bearish_rejection = upper_wick >= (1.5 * body) or (closed['Close'] < prev_closed['Low'])
‎
‎    verdict = "ATTENDRE"
‎    color = "#FFA500"
‎    action = "HOLD"
‎    motif = "Aucune configuration claire ou marché en consolidation."
‎    entry = None
‎    sl = None
‎    tp = None
‎
‎Condition ACHAT
‎    if trend_up and in_buy_zone and bullish_rejection and (40 <= closed['RSI'] <= 68):
‎        verdict = "ACHAT FORT (BUY)"
‎        color = "#00FF88"
‎        action = "BUY"
‎        motif = "Tendance haussière + Pullback sur zone EMA + Rejet haussier confirmé."
‎        entry = df.iloc[-1]['Open'] # Entrée à l'ouverture de la bougie actuelle
‎        sl = round(closed['Low'] - (0.5 * atr), 5)
‎        risk = entry - sl
‎        tp = round(entry + (2 * risk), 5) # Risk/Reward 1:2
‎
‎Condition VENTE
‎    elif trend_down and in_sell_zone and bearish_rejection and (32 <= closed['RSI'] <= 60):
‎        verdict = "VENTE FORTE (SELL)"
‎        color = "#FF3366"
‎        action = "SELL"
‎        motif = "Tendance baissière + Retest résistance EMA + Rejet baissier confirmé."
‎        entry = df.iloc[-1]['Open']
‎        sl = round(closed['High'] + (0.5 * atr), 5)
‎        risk = sl - entry
‎        tp = round(entry - (2 * risk), 5) # Risk/Reward 1:2
‎
‎    elif not trend_up and not trend_down:
‎        motif = "Marché sans tendance (Range / EMAs entrelacées). Risque élevé."
‎
‎    return {
‎        "verdict": verdict,
‎        "color": color,
‎        "action": action,
‎        "motif": motif,
‎        "entry": entry,
‎        "sl": sl,
‎        "tp": tp,
‎        "rsi": round(closed['RSI'], 2),
‎        "atr": round(atr, 5),
‎        "closed_time": closed['time']
‎    }
‎
‎--- AFFICHAGE PRINCIPAL ---
‎symbol_name, symbol_code = symbol_choice
‎tf_name, tf_seconds = timeframe_choice
‎
‎col_btn, col_info = st.columns([1, 4])
‎with col_btn:
‎    refresh = st.button("🔄 Rafraîchir l'analyse", use_container_width=True)
‎
‎Récupération des données
‎with st.spinner("Analyse du marché en cours via Deriv..."):
‎    df_candles = asyncio.run(fetch_deriv_candles(symbol_code, tf_seconds))
‎
‎if df_candles is not None and len(df_candles) > 60:
‎    res = analyze_market(df_candles)
‎    current_price = df_candles.iloc[-1]['Close']
‎
‎--- BANDEAU VERDICT ---
‎    st.markdown(f"""
‎        <div style='background-color: #1E222D; border-left: 8px solid {res["color"]}; padding: 20px; border-radius: 8px; margin-bottom: 20px;'>
‎            <div style='display: flex; justify-content: space-between; align-items: center;'>
‎                <div>
‎                    <h4 style='color: #888; margin:0;'>VERDICT ACTUEL ({symbol_name} - {tf_name})</h4>
‎                    <h1 style='color: {res["color"]}; margin: 5px 0 0 0;'>{res["verdict"]}</h1>
‎                    <p style='color: #DDD; margin: 8px 0 0 0;'><b>Raison :</b> {res["motif"]}</p>
‎                </div>
‎                <div style='text-align: right;'>
‎                    <span style='color: #888; font-size: 14px;'>Prix Direct</span>
‎                    <h2 style='color: #FFF; margin: 0;'>{current_price:.5f}</h2>
‎                    <small style='color: #777;'>Bougie analysée clôturée à {res["closed_time"].strftime("%H:%M UTC")}</small>
‎                </div>
‎            </div>
‎        </div>
‎    """, unsafe_allow_html=True)
‎
‎--- PARAMÈTRES D'EXÉCUTION (SI SIGNAL ACTIF) ---
‎    if res["action"] in ["BUY", "SELL"]:
‎        c1, c2, c3, c4 = st.columns(4)
‎        c1.metric("Prix d'Entrée Recommandé", f"{res['entry']:.5f}")
‎        c2.metric("Stop Loss (Sécurité)", f"{res['sl']:.5f}")
‎        c3.metric("Take Profit (Objectif)", f"{res['tp']:.5f}")
‎        c4.metric("Ratio Risque / Rendement", "1 : 2.0")
‎        st.info("💡 Conseil d'exécution : Ouvre MT5 et place ton ordre avec ces niveaux exacts.")
‎    else:
‎        st.caption("🟡 Aucun trade en cours : Préserver son capital fait partie du trading gagnant.")
‎
‎--- GRAPHIQUE INTERACTIF ---
‎    fig = go.Figure()
‎
‎Chandelier japonais
‎    fig.add_trace(go.Candlestick(
‎        x=df_candles['time'],
‎        open=df_candles['Open'],
‎        high=df_candles['High'],
‎        low=df_candles['Low'],
‎        close=df_candles['Close'],
‎        name="Prix"
‎    ))
‎
‎EMAs
‎    fig.add_trace(go.Scatter(x=df_candles['time'], y=df_candles['EMA20'], line=dict(color='#00D4B2', width=1.5), name="EMA 20 (Rapide)"))
‎    fig.add_trace(go.Scatter(x=df_candles['time'], y=df_candles['EMA50'], line=dict(color='#FF9900', width=1.5), name="EMA 50 (Tendance)"))
‎
‎    fig.update_layout(
‎        template="plotly_dark",
‎        xaxis_rangeslider_visible=False,
‎        height=450,
‎        margin=dict(l=10, r=10, t=10, b=10),
‎        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
‎    )
‎    st.plotly_chart(fig, use_container_width=True)
‎
‎Indicateurs annexes
‎    col_a, col_b = st.columns(2)
‎    col_a.metric("RSI (Momentum)", f"{res['rsi']}")
‎    col_b.metric("ATR (Volatilité en Pips)", f"{res['atr']:.5f}")
‎
‎else:
‎    st.error("Impossible de récupérer les flux de données Deriv. Réessaye dans quelques secondes.")
‎

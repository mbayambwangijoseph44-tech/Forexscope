import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timezone

# 1. Configuration immédiate
st.set_page_config(page_title="ForexScope Pro", page_icon="🎯", layout="wide")

# 2. Gestion du Journal
if "journal" not in st.session_state:
    st.session_state.journal = []

def ajouter_signal(paire, verdict, action, entry, sl, tp, tf, motif):
    if action in ["BUY", "SELL"]:
        st.session_state.journal.insert(0, {
            "Date/Heure": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Paire": paire,
            "TF": tf,
            "Verdict": verdict,
            "Entree": round(entry, 5) if entry else 0,
            "SL": round(sl, 5) if sl else 0,
            "TP": round(tp, 5) if tp else 0,
            "Motif": motif
        })

# 3. Sidebar
st.sidebar.title("🎯 FOREXSCOPE")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["🔍 Analyse en Direct", "📊 Scanner Multi-Paires", "📈 Graphique", "⚠️ Risque", "📓 Journal"])
st.sidebar.markdown("---")
st.sidebar.caption("v3.4 - Moteur Sécurisé")

PAIRS = {
    "AUD/USD": "AUDUSD=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CAD": "USDCAD=X",
    "EUR/GBP": "EURGBP=X",
    "USD/CHF": "USDCHF=X"
}

TIMEFRAMES = {
    "M15 (15 min)": {"interval": "15m", "period": "5d"},
    "M5 (5 min)": {"interval": "5m", "period": "2d"},
    "M30 (30 min)": {"interval": "30m", "period": "5d"},
    "H1 (1 heure)": {"interval": "1h", "period": "1mo"}
}

# 4. Téléchargement sécurisé et ultra-rapide
def get_forex_data(symbol, period, interval):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval, timeout=5)
        if df.empty or len(df) < 20:
            return None
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        return df
    except Exception:
        return None

# 5. Moteur d'analyse
def run_analysis(df):
    if df is None or len(df) < 20:
        return None

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # RSI
    diff = df['Close'].diff()
    gain = diff.clip(lower=0).rolling(14).mean()
    loss = (-diff.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())))
    atr = tr.rolling(14).mean().iloc[-2]

    # Bougie N-2 (clôturée)
    c = df.iloc[-2]
    prev = df.iloc[-3]
    body = abs(c['Close'] - c['Open'])
    lower_wick = min(c['Close'], c['Open']) - c['Low']
    upper_wick = c['High'] - max(c['Close'], c['Open'])

    trend_up = c['EMA20'] > c['EMA50']
    trend_down = c['EMA20'] < c['EMA50']

    verdict = "ATTENDRE"
    color = "#FFA500"
    action = "HOLD"
    motif = "En attente d'alignement de tendance et de rejet."
    entry = sl = tp = None

    # Condition ACHAT
    if trend_up and (lower_wick > 0.8 * body or c['Close'] > prev['High']) and (35 <= c['RSI'] <= 68):
        verdict = "ACHAT (BUY)"
        color = "#00FF88"
        action = "BUY"
        motif = "Tendance haussière + Rejet acheteur + RSI favorable."
        entry = df.iloc[-1]['Open']
        sl = round(c['Low'] - (0.4 * atr), 5)
        tp = round(entry + (1.8 * max(entry - sl, 0.0005)), 5)

    # Condition VENTE
    elif trend_down and (upper_wick > 0.8 * body or c['Close'] < prev['Low']) and (32 <= c['RSI'] <= 65):
        verdict = "VENTE (SELL)"
        color = "#FF3366"
        action = "SELL"
        motif = "Tendance baissière + Rejet vendeur + RSI favorable."
        entry = df.iloc[-1]['Open']
        sl = round(c['High'] + (0.4 * atr), 5)
        tp = round(entry - (1.8 * max(sl - entry, 0.0005)), 5)

    return {
        "verdict": verdict, "color": color, "action": action,
        "motif": motif, "entry": entry, "sl": sl, "tp": tp,
        "rsi": round(c['RSI'], 1), "price": df.iloc[-1]['Close'],
        "trend": "Haussière" if trend_up else "Baissière"
    }

# ============================================================
# PAGES DE L'APPLICATION
# ============================================================

# PAGE 1 : ANALYSE EN DIRECT
if page == "🔍 Analyse en Direct":
    st.markdown("## 🔍 Analyse Chirurgicale d'une Paire")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        p_name = st.selectbox("Paire", list(PAIRS.keys()), index=0)
    with col2:
        tf_name = st.selectbox("Unité de temps", list(TIMEFRAMES.keys()), index=0)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_calc = st.button("🚀 Analyser", use_container_width=True, type="primary")

    tf_cfg = TIMEFRAMES[tf_name]
    
    with st.spinner("Récupération du marché..."):
        df = get_forex_data(PAIRS[p_name], tf_cfg["period"], tf_cfg["interval"])
        res = run_analysis(df)

    if res:
        st.markdown(f"""
        <div style='background-color: #1E222D; border-left: 8px solid {res["color"]}; padding: 20px; border-radius: 8px; margin-bottom: 20px;'>
            <h1 style='color: {res["color"]}; margin: 0;'>{res["verdict"]}</h1>
            <p style='color: #CCC; margin: 8px 0 0 0;'><b>{p_name}</b> | {tf_name} | Prix direct : <b>{res['price']:.5f}</b></p>
            <p style='color: #AAA; margin: 5px 0 0 0;'><b>Explication :</b> {res["motif"]}</p>
        </div>
        """, unsafe_allow_html=True)

        if res["action"] in ["BUY", "SELL"]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prix d'Entrée", f"{res['entry']:.5f}")
            c2.metric("Stop Loss", f"{res['sl']:.5f}")
            c3.metric("Take Profit", f"{res['tp']:.5f}")
            c4.metric("Ratio R:R", "1 : 1.8")
            
            if st.button("📝 Sauvegarder dans le Journal", use_container_width=True):
                ajouter_signal(p_name, res["verdict"], res["action"], res["entry"], res["sl"], res["tp"], tf_name, res["motif"])
                st.success("Signal ajouté au journal !")
        else:
            st.info("Statut : En attente d'une opportunité claire.")

        ca, cb = st.columns(2)
        ca.metric("Tendance EMA", res["trend"])
        cb.metric("RSI (14)", res["rsi"])
    else:
        st.warning("Clique sur le bouton '🚀 Analyser' ou réessaye dans quelques secondes.")

# PAGE 2 : SCANNER MULTI-PAIRES
elif page == "📊 Scanner Multi-Paires":
    st.markdown("## 📊 Scanner Multi-Paires")
    st.markdown("---")
    tf_scan = st.selectbox("Unité de temps du scan", list(TIMEFRAMES.keys()), index=0)
    tf_cfg = TIMEFRAMES[tf_scan]

    if st.button("🚀 Lancer le Scan Global", use_container_width=True, type="primary"):
        rows = []
        bar = st.progress(0)
        for i, (name, sym) in enumerate(PAIRS.items()):
            df = get_forex_data(sym, tf_cfg["period"], tf_cfg["interval"])
            res = run_analysis(df)
            if res:
                if res["action"] in ["BUY", "SELL"]:
                    ajouter_signal(name, res["verdict"], res["action"], res["entry"], res["sl"], res["tp"], tf_scan, res["motif"])
                rows.append({
                    "Paire": name, "Verdict": res["verdict"], "Prix": f"{res['price']:.5f}",
                    "Tendance": res["trend"], "RSI": res["rsi"], "Raison": res["motif"]
                })
            bar.progress((i + 1) / len(PAIRS))
        
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# PAGE 3 : GRAPHIQUE
elif page == "📈 Graphique":
    st.markdown("## 📈 Graphique Chandelier")
    st.markdown("---")
    p_name = st.selectbox("Paire", list(PAIRS.keys()), index=0)
    tf_name = st.selectbox("Unité de temps", list(TIMEFRAMES.keys()), index=0)
    tf_cfg = TIMEFRAMES[tf_name]

    df = get_forex_data(PAIRS[p_name], tf_cfg["period"], tf_cfg["interval"])
    if df is not None and len(df) > 10:
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Prix"))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#00D4B2', width=1.5), name="EMA 20"))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#FF9900', width=1.5), name="EMA 50"))
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

# PAGE 4 : RISQUE
elif page == "⚠️ Risque":
    st.markdown("## ⚠️ Calculateur de Lot MT5")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    cap = c1.number_input("Capital ($)", value=1000.0, step=100.0)
    risk = c2.number_input("Risque (%)", value=1.0, step=0.5)
    sl_p = c3.number_input("Stop Loss (pips)", value=20.0, step=1.0)
    
    montant = cap * (risk / 100)
    lot = montant / (sl_p * 10.0)
    
    r1, r2 = st.columns(2)
    r1.metric("Montant Risqué", f"${montant:.2f}")
    r2.metric("Taille de Lot", f"{lot:.2f} lots")

# PAGE 5 : JOURNAL
elif page == "📓 Journal":
    st.markdown("## 📓 Journal des Signaux")
    st.markdown("---")
    if len(st.session_state.journal) == 0:
        st.info("📭 Aucun signal pour le moment.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.journal), use_container_width=True, hide_index=True)
        if st.button("🗑️ Vider le journal"):
            st.session_state.journal = []
            st.rerun()

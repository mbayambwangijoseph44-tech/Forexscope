import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timezone

st.set_page_config(page_title="ForexScope Pro", page_icon="🎯", layout="wide")

# ============================================================
# INITIALISATION DU JOURNAL
# ============================================================
if "journal" not in st.session_state:
    st.session_state.journal = []

def ajouter_au_journal(paire, verdict, action, entry, sl, tp, timeframe, motif, score):
    if action in ["BUY", "SELL"]:
        for s in st.session_state.journal[:5]:
            if s["Paire"] == paire and s["Verdict"] == verdict:
                return
        signal = {
            "Date/Heure": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Paire": paire,
            "Timeframe": timeframe,
            "Verdict": verdict,
            "Score": f"{score}%",
            "Entree": round(entry, 5) if entry else 0,
            "Stop Loss": round(sl, 5) if sl else 0,
            "Take Profit": round(tp, 5) if tp else 0,
            "Motif": motif,
            "Statut": "Actif"
        }
        st.session_state.journal.insert(0, signal)

# ============================================================
# NAVIGATION LATÉRALE
# ============================================================
st.sidebar.markdown("# 🎯 FOREXSCOPE")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Menu",
    ["📊 Tableau de bord", "🔍 Analyse Paire", "📈 Graphique", "⚠️ Risque", "📓 Journal"]
)
st.sidebar.markdown("---")

st.sidebar.markdown("### 🎛️ Mode")
sensibilite = st.sidebar.selectbox(
    "Sensibilité des signaux",
    ["Modéré (Recommandé)", "Dynamique (Plus de signaux)", "Strict (Moins de signaux)"],
    index=0
)

# ============================================================
# PAIRES ET TIMEFRAMES
# ============================================================
PAIRS = {
    "AUD/USD": "AUDUSD=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CAD": "USDCAD=X",
    "EUR/GBP": "EURGBP=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CHF": "USDCHF=X",
}

TIMEFRAMES = {
    "M5 (5 min)": {"interval": "5m", "period": "5d"},
    "M15 (15 min)": {"interval": "15m", "period": "1mo"},
    "M30 (30 min)": {"interval": "30m", "period": "1mo"},
    "H1 (1 heure)": {"interval": "1h", "period": "3mo"},
}

# ============================================================
# RÉCUPÉRATION RAPIDE ET SÉCURISÉE DES DONNÉES
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, timeout=7)
        if df is None or df.empty or len(df) < 30:
            return None
        # Nettoyage colonnes
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        return df
    except Exception:
        return None

# ============================================================
# MOTEUR D'ANALYSE (ZERO REPAINT)
# ============================================================
def analyze_data(df, mode):
    if df is None or len(df) < 30:
        return None

    df = df.copy()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = tr.rolling(14).mean()

    # Analyse stricte sur bougie N-2
    c = df.iloc[-2]
    prev = df.iloc[-3]
    atr = c['ATR'] if not np.isnan(c['ATR']) else (c['High'] - c['Low'])

    body = abs(c['Close'] - c['Open'])
    upper_wick = c['High'] - max(c['Close'], c['Open'])
    lower_wick = min(c['Close'], c['Open']) - c['Low']

    score_buy = 0
    score_sell = 0
    motifs = []

    # Tendance
    if c['EMA20'] > c['EMA50']:
        score_buy += 30
    elif c['EMA20'] < c['EMA50']:
        score_sell += 30

    # Contact EMA / Pullback
    dist_ema = abs(c['Close'] - c['EMA20']) / (c['Close'] + 1e-9)
    if dist_ema < 0.003:
        score_buy += 20
        score_sell += 20
        motifs.append("Contact EMA20 (Pullback)")

    # Rejets
    if lower_wick > (0.8 * body):
        score_buy += 25
        motifs.append("Rejet acheteur (mèche basse)")
    if upper_wick > (0.8 * body):
        score_sell += 25
        motifs.append("Rejet vendeur (mèche haute)")

    # RSI
    if 40 <= c['RSI'] <= 65:
        score_buy += 15
        score_sell += 15

    # Seuil
    seuil = 55 if "Dynamique" in mode else (75 if "Strict" in mode else 65)

    verdict = "ATTENDRE"
    color = "#FFA500"
    action = "HOLD"
    score = 0
    entry = sl = tp = None

    if score_buy >= seuil and score_buy > score_sell:
        verdict = f"🟢 ACHAT ({score_buy}%)"
        color = "#00FF88"
        action = "BUY"
        score = score_buy
        entry = df.iloc[-1]['Open']
        sl = round(c['Low'] - (0.4 * atr), 5)
        risk = max(entry - sl, 0.0005)
        tp = round(entry + (1.8 * risk), 5)
        motif = " + ".join(motifs) if motifs else "Tendance haussière confirmée."

    elif score_sell >= seuil and score_sell > score_buy:
        verdict = f"🔴 VENTE ({score_sell}%)"
        color = "#FF3366"
        action = "SELL"
        score = score_sell
        entry = df.iloc[-1]['Open']
        sl = round(c['High'] + (0.4 * atr), 5)
        risk = max(sl - entry, 0.0005)
        tp = round(entry - (1.8 * risk), 5)
        motif = " + ".join(motifs) if motifs else "Tendance baissière confirmée."

    else:
        score = max(score_buy, score_sell)
        motif = f"Score ({score}% / {seuil}% requis). Configuration incomplète."

    return {
        "verdict": verdict, "color": color, "action": action,
        "motif": motif, "entry": entry, "sl": sl, "tp": tp,
        "score": score, "rsi": round(c['RSI'], 1),
        "atr": round(atr, 5),
        "trend": "Haussière" if c['EMA20'] > c['EMA50'] else "Baissière",
        "price": df.iloc[-1]['Close']
    }

# ============================================================
# PAGE 1 : TABLEAU DE BORD
# ============================================================
if page == "📊 Tableau de bord":
    st.markdown("# 📊 Tableau de Bord")
    st.markdown("---")

    c1, c2, c3 = st.columns([2, 2, 1])
    c1.success("🟢 Système Prêt")
    c2.info(f"🕐 UTC : {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    if c3.button("🔄 Rafraîchir", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### Scanner Rapide")
    tf_scan = st.selectbox("Unité de temps", list(TIMEFRAMES.keys()), index=1)
    tf_cfg = TIMEFRAMES[tf_scan]

    if st.button("🚀 Lancer le Scan", use_container_width=True, type="primary"):
        results = []
        bar = st.progress(0)
        for i, (name, sym) in enumerate(PAIRS.items()):
            df = fetch_data(sym, tf_cfg["period"], tf_cfg["interval"])
            res = analyze_data(df, sensibilite)
            if res:
                if res["action"] in ["BUY", "SELL"]:
                    ajouter_au_journal(
                        name, res["verdict"], res["action"],
                        res["entry"], res["sl"], res["tp"], tf_scan, res["motif"], res["score"]
                    )
                results.append({
                    "Paire": name,
                    "Verdict": res["verdict"],
                    "Prix": f"{res['price']:.5f}",
                    "Tendance": res["trend"],
                    "RSI": res["rsi"],
                    "Raison": res["motif"]
                })
            bar.progress((i + 1) / len(PAIRS))

        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        else:
            st.warning("Données en cours de synchronisation. Réessaie dans quelques secondes.")

# ============================================================
# PAGE 2 : ANALYSE PAIRE
# ============================================================
elif page == "🔍 Analyse Paire":
    st.markdown("# 🔍 Analyse en Direct")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        pair_name = st.selectbox("Paire", list(PAIRS.keys()), index=0)
    with col2:
        tf_name = st.selectbox("Unité de temps", list(TIMEFRAMES.keys()), index=1)

    tf_cfg = TIMEFRAMES[tf_name]
    df = fetch_data(PAIRS[pair_name], tf_cfg["period"], tf_cfg["interval"])
    res = analyze_data(df, sensibilite)

    if res:
        st.markdown(f"""
        <div style='background-color: #1E222D; border-left: 8px solid {res["color"]}; padding: 18px; border-radius: 8px; margin-bottom: 15px;'>
            <h1 style='color: {res["color"]}; margin: 0;'>{res["verdict"]}</h1>
            <p style='color: #CCC; margin: 8px 0 0 0;'><b>{pair_name}</b> | {tf_name} | Prix direct : <b>{res['price']:.5f}</b></p>
            <p style='color: #AAA; margin: 5px 0 0 0;'><b>Raison :</b> {res["motif"]}</p>
        </div>
        """, unsafe_allow_html=True)

        if res["action"] in ["BUY", "SELL"]:
            st.markdown("### 📋 Ordre MT5 Recommandé")
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Prix Entrée", f"{res['entry']:.5f}")
            x2.metric("Stop Loss", f"{res['sl']:.5f}")
            x3.metric("Take Profit", f"{res['tp']:.5f}")
            x4.metric("Ratio R:R", "1 : 1.8")

            if st.button("📝 Sauvegarder dans le Journal", use_container_width=True, type="primary"):
                ajouter_au_journal(
                    pair_name, res["verdict"], res["action"],
                    res["entry"], res["sl"], res["tp"], tf_name, res["motif"], res["score"]
                )
                st.success("Signal ajouté au journal !")
        else:
            st.info("🟡 Aucune entrée immédiate. Passe en mode 'Dynamique' dans le menu latéral si tu souhaites plus d'opportunités.")

        c_a, c_b = st.columns(2)
        c_a.metric("Tendance", res["trend"])
        c_b.metric("RSI (14)", res["rsi"])
    else:
        st.warning("Chargement des données... Si cela persiste, clique sur Rafraîchir.")

# ============================================================
# PAGE 3 : GRAPHIQUE
# ============================================================
elif page == "📈 Graphique":
    st.markdown("# 📈 Graphique Chandelier")
    st.markdown("---")
    pair_name = st.selectbox("Paire", list(PAIRS.keys()), index=0)
    tf_name = st.selectbox("Unité de temps", list(TIMEFRAMES.keys()), index=1)

    tf_cfg = TIMEFRAMES[tf_name]
    df = fetch_data(PAIRS[pair_name], tf_cfg["period"], tf_cfg["interval"])

    if df is not None and len(df) > 20:
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name="Prix",
            increasing_line_color='#00FF88', decreasing_line_color='#FF3366'
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#00D4B2', width=1.5), name="EMA 20"))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#FF9900', width=1.5), name="EMA 50"))
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 4 : RISQUE
# ============================================================
elif page == "⚠️ Risque":
    st.markdown("# ⚠️ Calculateur de Lot MT5")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    capital = c1.number_input("Capital ($)", value=1000.0, step=100.0)
    risk_pct = c2.number_input("Risque max (%)", value=1.0, max_value=5.0, step=0.5)
    sl_pips = c3.number_input("Stop Loss (pips)", value=20.0, step=1.0)

    risk_amount = capital * (risk_pct / 100)
    lot_size = risk_amount / (sl_pips * 10.0)

    r1, r2 = st.columns(2)
    r1.metric("Montant Risqué", f"${risk_amount:.2f}")
    r2.metric("Lot Recommandé", f"{lot_size:.2f} lots")

# ============================================================
# PAGE 5 : JOURNAL
# ============================================================
elif page == "📓 Journal":
    st.markdown("# 📓 Journal des Signaux")
    st.markdown("---")
    if len(st.session_state.journal) == 0:
        st.info("📭 Aucun signal enregistré. Lance un scan ou une analyse pour en ajouter.")
    else:
        df_j = pd.DataFrame(st.session_state.journal)
        st.dataframe(df_j, use_container_width=True, hide_index=True)
        if st.button("🗑️ Vider le journal"):
            st.session_state.journal = []
            st.rerun()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timezone

st.set_page_config(page_title="ForexScope Pro v3.1", page_icon="🎯", layout="wide")

# ============================================================
# INITIALISATION DU JOURNAL
# ============================================================
if "journal" not in st.session_state:
    st.session_state.journal = []

def ajouter_au_journal(paire, verdict, action, entry, sl, tp, timeframe, motif):
    if action in ["BUY", "SELL"]:
        # Eviter les doublons recents
        for s in st.session_state.journal[:3]:
            if s["Paire"] == paire and s["Verdict"] == verdict:
                return
        signal = {
            "Date/Heure": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Paire": paire,
            "Timeframe": timeframe,
            "Verdict": verdict,
            "Entree": round(entry, 5) if entry else 0,
            "Stop Loss": round(sl, 5) if sl else 0,
            "Take Profit": round(tp, 5) if tp else 0,
            "Motif": motif,
            "Statut": "Actif"
        }
        st.session_state.journal.insert(0, signal)

# ============================================================
# NAVIGATION
# ============================================================
st.sidebar.markdown("# 🎯 FOREXSCOPE")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Tableau de bord", "🔍 Analyse", "📈 Graphique", "⚠️ Risque", "📓 Journal", "⚙️ Parametres"]
)
st.sidebar.markdown("---")

# Session actuelle
heure_utc = datetime.now(timezone.utc).hour
if 7 <= heure_utc < 16:
    session_status = "🟢 Londres (Active)"
elif 12 <= heure_utc < 20:
    session_status = "🟢 New York (Active)"
else:
    session_status = "🔴 Asie / Hors Session (Risque eleve)"

st.sidebar.caption(f"Session : {session_status}")
st.sidebar.caption(f"📓 Journal : {len(st.session_state.journal)} signaux")
st.sidebar.caption("v3.1 - Filtre Multi-Timeframe & Session")

# ============================================================
# CONFIGURATION PAIRES
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

def tester_connexion_yahoo():
    try:
        test = yf.Ticker("AUDUSD=X")
        hist = test.history(period="1d", interval="1h")
        if hist.empty:
            return False, "Aucune donnee recue"
        return True, "Connecte"
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=60)
def fetch_data(yahoo_symbol, period, interval):
    try:
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception:
        return None

# ============================================================
# MOTEUR D'ANALYSE v3.1 AVEC FILTRE MTF ET SESSION
# ============================================================
def analyze_market_v3(df_ltf, df_htf=None, bypass_session=False):
    if df_ltf is None or len(df_ltf) < 60:
        return None

    df = df_ltf.copy()
    
    # Indicateurs LTF
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

    # ADX
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = df['ATR']
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-9))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(14).mean()

    # Bougie cloturee N-2 (ZERO REPAINT)
    c = df.iloc[-2]
    prev = df.iloc[-3]
    atr_val = c['ATR']

    body = abs(c['Close'] - c['Open'])
    upper_wick = c['High'] - max(c['Close'], c['Open'])
    lower_wick = min(c['Close'], c['Open']) - c['Low']

    # --- 1. FILTRE HORAIRE SESSION ---
    now_hour = datetime.now(timezone.utc).hour
    is_liquid_session = (7 <= now_hour < 20) # Entre 07h00 et 20h00 UTC
    
    # --- 2. FILTRE TENDANCE SUPÉRIEURE (HTF H1) ---
    htf_trend = "Neutre"
    if df_htf is not None and len(df_htf) > 50:
        htf_ema20 = df_htf['Close'].ewm(span=20, adjust=False).mean().iloc[-2]
        htf_ema50 = df_htf['Close'].ewm(span=50, adjust=False).mean().iloc[-2]
        if htf_ema20 > htf_ema50:
            htf_trend = "Haussiere"
        elif htf_ema20 < htf_ema50:
            htf_trend = "Baissiere"

    # Tendance LTF
    trend_up = c['EMA20'] > c['EMA50'] and c['Close'] > c['EMA50']
    trend_down = c['EMA20'] < c['EMA50'] and c['Close'] < c['EMA50']
    is_ranging = c['ADX'] < 22 if not np.isnan(c['ADX']) else False

    # Zone de valeur stricte
    in_buy_zone = c['Low'] <= c['EMA20'] and c['Close'] >= c['EMA50']
    in_sell_zone = c['High'] >= c['EMA20'] and c['Close'] <= c['EMA50']

    # Rejet STRICT (mèche >= 1.8x le corps pour éliminer les micro-mèches)
    bullish_reject = (lower_wick >= (1.8 * body)) or (c['Close'] > prev['High'] and c['Close'] > c['Open'] and body > 0.6 * (c['High'] - c['Low']))
    bearish_reject = (upper_wick >= (1.8 * body)) or (c['Close'] < prev['Low'] and c['Close'] < c['Open'] and body > 0.6 * (c['High'] - c['Low']))

    verdict = "ATTENDRE"
    color = "#FFA500"
    action = "HOLD"
    motif = "Aucune configuration valide a haute probabilite."
    entry = sl = tp = None
    score = 0

    # APPLICATION DES FILTRES DE SECURITE
    if not is_liquid_session and not bypass_session:
        motif = "🛑 Hors session liquide (Session Asie). Risque eleve de faux signaux."
    elif is_ranging:
        motif = "Marche en range / consolidation (ADX < 22). Pas de tendance claire."
    
    # SIGNAL ACHAT
    elif trend_up and (htf_trend in ["Haussiere", "Neutre"]) and in_buy_zone and bullish_reject and (42 <= c['RSI'] <= 65):
        verdict = "ACHAT FORT (BUY)"
        color = "#00FF88"
        action = "BUY"
        motif = f"Tendance LTF + HTF ({htf_trend}) + Pullback propre sur EMA + Rejet haussier net."
        entry = df.iloc[-1]['Open']
        sl = round(c['Low'] - (0.4 * atr_val), 5)
        risk = entry - sl
        tp = round(entry + (2 * risk), 5) if risk > 0 else round(entry + (2 * atr_val), 5)
        score = 85

    # SIGNAL VENTE
    elif trend_down and (htf_trend in ["Baissiere", "Neutre"]) and in_sell_zone and bearish_reject and (35 <= c['RSI'] <= 58):
        verdict = "VENTE FORTE (SELL)"
        color = "#FF3366"
        action = "SELL"
        motif = f"Tendance LTF + HTF ({htf_trend}) + Retest propre EMA + Rejet baissier net."
        entry = df.iloc[-1]['Open']
        sl = round(c['High'] + (0.4 * atr_val), 5)
        risk = sl - entry
        tp = round(entry - (2 * risk), 5) if risk > 0 else round(entry - (2 * atr_val), 5)
        score = -85

    elif trend_up and htf_trend == "Baissiere":
        motif = "⚠️ Signal haussier M15 bloque : la tendance H1 est fortement baissiere."
    elif trend_down and htf_trend == "Haussiere":
        motif = "⚠️ Signal baissier M15 bloque : la tendance H1 est fortement haussiere."
    elif trend_up and not in_buy_zone:
        motif = "Tendance haussiere mais prix trop eloigne des EMAs. Attendre un pullback."
    elif trend_down and not in_sell_zone:
        motif = "Tendance baissiere mais prix trop eloigne des EMAs. Attendre un retest."

    return {
        "verdict": verdict, "color": color, "action": action,
        "motif": motif, "entry": entry, "sl": sl, "tp": tp,
        "score": score, "rsi": round(c['RSI'], 1),
        "atr": round(atr_val, 5),
        "adx": round(c['ADX'], 1) if not np.isnan(c['ADX']) else 0,
        "htf_trend": htf_trend,
        "closed_time": str(df.index[-2])
    }

# ============================================================
# PAGE 1 : TABLEAU DE BORD
# ============================================================
if page == "📊 Tableau de bord":
    st.markdown("# 📊 Tableau de Bord")
    st.markdown("---")

    col_statut, col_heure, col_session, col_refresh = st.columns([2, 2, 2, 1])

    with col_statut:
        connected, msg = tester_connexion_yahoo()
        if connected:
            st.success("🟢 Yahoo Finance : Connecte")
        else:
            st.error("🔴 Deconnecte")

    with col_heure:
        st.info(f"🕐 {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

    with col_session:
        if "🟢" in session_status:
            st.success(f"Session : {session_status}")
        else:
            st.warning(f"Session : {session_status}")

    with col_refresh:
        if st.button("🔄 Actualiser", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("### Scanner Multi-Paires Filtré")
    tf_scan = st.selectbox("Unite de temps LTF", list(TIMEFRAMES.keys()), index=1)
    tf_cfg = TIMEFRAMES[tf_scan]

    bypass = st.checkbox("Ignorer le filtre de session (Mode Test Nocturne)", value=False)

    if st.button("🚀 Scanner les 8 paires majeures", use_container_width=True, type="primary"):
        results = []
        progress = st.progress(0)
        for i, (name, yahoo_sym) in enumerate(PAIRS.items()):
            df_ltf = fetch_data(yahoo_sym, tf_cfg["period"], tf_cfg["interval"])
            df_htf = fetch_data(yahoo_sym, "3mo", "1h") # H1 comme référence HTF
            res = analyze_market_v3(df_ltf, df_htf, bypass_session=bypass)
            if res:
                if res["action"] in ["BUY", "SELL"]:
                    ajouter_au_journal(
                        name, res["verdict"], res["action"],
                        res["entry"], res["sl"], res["tp"], tf_scan, res["motif"]
                    )
                results.append({
                    "Paire": name,
                    "Verdict": res["verdict"],
                    "Tendance H1": res["htf_trend"],
                    "RSI": res["rsi"],
                    "ADX": res["adx"],
                    "Raison": res["motif"]
                })
            progress.progress((i + 1) / len(PAIRS))

        if results:
            st.markdown("---")
            for r in results:
                if "ACHAT" in r["Verdict"]:
                    st.success(f"🟢 **{r['Paire']}** | {r['Verdict']} | H1: {r['Tendance H1']} | RSI: {r['RSI']} | {r['Raison']}")
                elif "VENTE" in r["Verdict"]:
                    st.error(f"🔴 **{r['Paire']}** | {r['Verdict']} | H1: {r['Tendance H1']} | RSI: {r['RSI']} | {r['Raison']}")
                else:
                    st.warning(f"🟡 **{r['Paire']}** | {r['Verdict']} | H1: {r['Tendance H1']} | {r['Raison']}")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

# ============================================================
# PAGE 2 : ANALYSE DÉTAILLÉE
# ============================================================
elif page == "🔍 Analyse":
    st.markdown("# 🔍 Analyse Détaillée Multi-Timeframe")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        pair_name = st.selectbox("Paire", list(PAIRS.keys()), index=0)
    with col2:
        tf_name = st.selectbox("Unité de temps", list(TIMEFRAMES.keys()), index=1)
    with col3:
        bypass = st.checkbox("Bypass Session", value=False)
        if st.button("🔄 Analyser", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    tf_cfg = TIMEFRAMES[tf_name]
    yahoo_sym = PAIRS[pair_name]

    with st.spinner("Analyse LTF + Tendance H1 en cours..."):
        df_ltf = fetch_data(yahoo_sym, tf_cfg["period"], tf_cfg["interval"])
        df_htf = fetch_data(yahoo_sym, "3mo", "1h")
        res = analyze_market_v3(df_ltf, df_htf, bypass_session=bypass)

    if res:
        current_price = df_ltf.iloc[-1]['Close']

        st.markdown(f"""
        <div style='background-color: #1E222D; border-left: 8px solid {res["color"]}; padding: 20px; border-radius: 8px;'>
            <h1 style='color: {res["color"]}; margin: 0;'>{res["verdict"]}</h1>
            <p style='color: #CCC; margin: 8px 0 0 0;'><b>{pair_name}</b> | {tf_name} | Prix: {current_price:.5f} | Tendance de fond H1 : <b>{res['htf_trend']}</b></p>
            <p style='color: #AAA; margin: 5px 0 0 0;'><b>Explication :</b> {res["motif"]}</p>
        </div>
        """, unsafe_allow_html=True)

        if res["action"] in ["BUY", "SELL"]:
            st.markdown("### 📋 Paramètres d'Exécution MT5")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prix Entrée", f"{res['entry']:.5f}")
            c2.metric("Stop Loss", f"{res['sl']:.5f}")
            c3.metric("Take Profit", f"{res['tp']:.5f}")
            c4.metric("Ratio R:R", "1 : 2.0")

            if st.button("📝 Enregistrer dans le Journal", use_container_width=True, type="primary"):
                ajouter_au_journal(
                    pair_name, res["verdict"], res["action"],
                    res["entry"], res["sl"], res["tp"], tf_name, res["motif"]
                )
                st.success("Enregistré !")
        else:
            st.info("💡 **Conseil Pro :** Ne pas être en position est aussi une position. Attends que toutes les conditions soient alignées.")

        st.markdown("### 📊 Données Techniques")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tendance H1 (Majeure)", res["htf_trend"])
        c2.metric("RSI (14)", res["rsi"])
        c3.metric("ADX (Force tendance)", res["adx"])
        c4.metric("ATR (Volatilité)", res["atr"])

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
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=550)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 4 : RISQUE
# ============================================================
elif page == "⚠️ Risque":
    st.markdown("# ⚠️ Calculateur de Lot Strict")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    capital = c1.number_input("Capital ($)", value=1000.0, step=100.0)
    risk_pct = c2.number_input("Risque max par trade (%)", value=1.0, max_value=3.0, step=0.5)
    sl_pips = c3.number_input("Stop Loss (pips)", value=20.0, step=1.0)

    risk_amount = capital * (risk_pct / 100)
    lot_size = risk_amount / (sl_pips * 10.0)

    rc1, rc2 = st.columns(2)
    rc1.metric("Montant Risqué", f"${risk_amount:.2f}")
    rc2.metric("Taille du Lot Recommandée", f"{lot_size:.2f} lots")
    st.info("💡 Ne trade jamais plus de 1% à 2% de risque par trade.")

# ============================================================
# PAGE 5 : JOURNAL
# ============================================================
elif page == "📓 Journal":
    st.markdown("# 📓 Journal des Signaux")
    st.markdown("---")
    if len(st.session_state.journal) == 0:
        st.info("📭 Aucun signal enregistré.")
    else:
        df_j = pd.DataFrame(st.session_state.journal)
        st.dataframe(df_j, use_container_width=True, hide_index=True)
        if st.button("🗑️ Vider le journal"):
            st.session_state.journal = []
            st.rerun()

# ============================================================
# PAGE 6 : PARAMÈTRES
# ============================================================
elif page == "⚙️ Parametres":
    st.markdown("# ⚙️ Paramètres")
    st.markdown("### Filtres actifs dans cette version :")
    st.markdown("""
    - ✅ **Kill-Zone Session :** Bloque les trades entre 21h et 07h UTC (Session Asiatique).
    - ✅ **Validation HTF H1 :** Interdit d'aller contre la tendance H1.
    - ✅ **Rejet Minimum :** Mèche >= 1.8x le corps de la bougie.
    - ✅ **Zéro Repaint :** Calcul strict sur bougie N-2.
    """)

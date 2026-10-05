import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timezone

st.set_page_config(page_title="ForexScope Pro v3.2", page_icon="🎯", layout="wide")

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
# SIDEBAR
# ============================================================
st.sidebar.markdown("# 🎯 FOREXSCOPE")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Tableau de bord", "🔍 Analyse", "📈 Graphique", "⚠️ Risque", "📓 Journal", "⚙️ Parametres"]
)
st.sidebar.markdown("---")

st.sidebar.markdown("### 🎛️ Sensibilité des Signaux")
sensibilite = st.sidebar.select_slider(
    "Mode d'analyse",
    options=["Dynamique (Plus d'entrées)", "Modéré (Équilibré)", "Strict (Haute Précision)"],
    value="Modéré (Équilibré)"
)

heure_utc = datetime.now(timezone.utc).hour
if 7 <= heure_utc < 16:
    session_status = "🟢 Londres (Active)"
elif 12 <= heure_utc < 20:
    session_status = "🟢 New York (Active)"
else:
    session_status = "🟡 Asie / Nuit (Volatilité réduite)"

st.sidebar.markdown("---")
st.sidebar.caption(f"Session : {session_status}")
st.sidebar.caption(f"📓 Signaux dans le journal : {len(st.session_state.journal)}")

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

@st.cache_data(ttl=45)
def fetch_data(yahoo_symbol, period, interval):
    try:
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty or len(df) < 30:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        return df
    except Exception:
        return None

# ============================================================
# MOTEUR DE DÉCISION PONDÉRÉ (ZERO REPAINT)
# ============================================================
def analyze_market_v32(df_ltf, mode="Modéré (Équilibré)"):
    if df_ltf is None or len(df_ltf) < 40:
        return None

    df = df_ltf.copy()

    # Moyennes Mobiles
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

    # Analyse sur bougie clôturée N-2
    c = df.iloc[-2]
    prev = df.iloc[-3]
    atr_val = c['ATR'] if not np.isnan(c['ATR']) else (c['High'] - c['Low'])

    body = abs(c['Close'] - c['Open'])
    upper_wick = c['High'] - max(c['Close'], c['Open'])
    lower_wick = min(c['Close'], c['Open']) - c['Low']

    # --- CALCUL DU SCORE D'ACHAT ET DE VENTE ---
    score_buy = 0
    score_sell = 0
    reasons = []

    # 1. Tendance EMA
    if c['EMA20'] > c['EMA50']:
        score_buy += 25
        if c['Close'] > c['EMA20']:
            score_buy += 10
    elif c['EMA20'] < c['EMA50']:
        score_sell += 25
        if c['Close'] < c['EMA20']:
            score_sell += 10

    # 2. Zone de Retest / Pullback
    dist_to_ema20 = abs(c['Close'] - c['EMA20']) / (c['Close'] + 1e-9)
    if dist_to_ema20 < 0.0025:  # Prix proche de l'EMA
        score_buy += 20
        score_sell += 20
        reasons.append("Prix en zone de contact EMA (Pullback)")

    # 3. Action des prix & Rejets
    if lower_wick > body:
        score_buy += 25
        reasons.append("Rejet par le bas (Mèche d'acheteurs)")
    if c['Close'] > prev['High']:
        score_buy += 15
        reasons.append("Cassure haussière de la bougie précédente")

    if upper_wick > body:
        score_sell += 25
        reasons.append("Rejet par le haut (Mèche de vendeurs)")
    if c['Close'] < prev['Low']:
        score_sell += 15
        reasons.append("Cassure baissière de la bougie précédente")

    # 4. Filtre RSI
    rsi_val = c['RSI']
    if 40 <= rsi_val <= 65:
        score_buy += 10
        score_sell += 10
    elif rsi_val < 30:
        score_buy += 15
        reasons.append("RSI en survente (Rebond potentiel)")
    elif rsi_val > 70:
        score_sell += 15
        reasons.append("RSI en surachat (Repli potentiel)")

    # Seuil selon la sensibilité
    if "Dynamique" in mode:
        seuil = 55
    elif "Strict" in mode:
        seuil = 80
    else:
        seuil = 65

    verdict = "ATTENDRE"
    color = "#FFA500"
    action = "HOLD"
    final_score = 0
    entry = sl = tp = None

    if score_buy >= seuil and score_buy > score_sell:
        verdict = f"🟢 ACHAT ({score_buy}%)"
        color = "#00FF88"
        action = "BUY"
        final_score = score_buy
        entry = df.iloc[-1]['Open']
        sl = round(c['Low'] - (0.4 * atr_val), 5)
        risk = max(entry - sl, 0.0005)
        tp = round(entry + (1.8 * risk), 5)
        motif = " + ".join(reasons) if reasons else "Signal d'achat haussier validé."

    elif score_sell >= seuil and score_sell > score_buy:
        verdict = f"🔴 VENTE ({score_sell}%)"
        color = "#FF3366"
        action = "SELL"
        final_score = score_sell
        entry = df.iloc[-1]['Open']
        sl = round(c['High'] + (0.4 * atr_val), 5)
        risk = max(sl - entry, 0.0005)
        tp = round(entry - (1.8 * risk), 5)
        motif = " + ".join(reasons) if reasons else "Signal de vente baissier validé."

    else:
        final_score = max(score_buy, score_sell)
        motif = f"Score insuffisant ({final_score}% / {seuil}% requis). Marché en attente de configuration."

    return {
        "verdict": verdict, "color": color, "action": action,
        "motif": motif, "entry": entry, "sl": sl, "tp": tp,
        "score": final_score, "rsi": round(rsi_val, 1),
        "atr": round(atr_val, 5),
        "trend": "Haussière" if c['EMA20'] > c['EMA50'] else "Baissière",
        "closed_time": str(df.index[-2])
    }

# ============================================================
# PAGE 1 : TABLEAU DE BORD
# ============================================================
if page == "📊 Tableau de bord":
    st.markdown("# 📊 Tableau de Bord ForexScope")
    st.markdown("---")

    c_stat, c_time, c_sens, c_btn = st.columns([2, 2, 2, 1])
    with c_stat:
        st.success("🟢 Flux Yahoo Finance : Actif")
    with c_time:
        st.info(f"🕐 UTC : {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
    with c_sens:
        st.caption(f"Mode actuel : **{sensibilite}**")
    with c_btn:
        if st.button("🔄 Rafraîchir", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("### Scanner des Opportunités en Direct")
    tf_scan = st.selectbox("Unité de temps", list(TIMEFRAMES.keys()), index=1)
    tf_cfg = TIMEFRAMES[tf_scan]

    if st.button("🚀 Scanner toutes les paires", use_container_width=True, type="primary"):
        results = []
        progress = st.progress(0)
        for i, (name, y_sym) in enumerate(PAIRS.items()):
            df = fetch_data(y_sym, tf_cfg["period"], tf_cfg["interval"])
            res = analyze_market_v32(df, mode=sensibilite)
            if res:
                if res["action"] in ["BUY", "SELL"]:
                    ajouter_au_journal(
                        name, res["verdict"], res["action"],
                        res["entry"], res["sl"], res["tp"], tf_scan, res["motif"], res["score"]
                    )
                results.append({
                    "Paire": name,
                    "Verdict": res["verdict"],
                    "Tendance": res["trend"],
                    "RSI": res["rsi"],
                    "Score": f"{res['score']}%",
                    "Raison": res["motif"]
                })
            progress.progress((i + 1) / len(PAIRS))

        if results:
            st.markdown("---")
            for r in results:
                if "ACHAT" in r["Verdict"]:
                    st.success(f"🟢 **{r['Paire']}** | **{r['Verdict']}** | Tendance : {r['Tendance']} | RSI : {r['RSI']} | {r['Raison']}")
                elif "VENTE" in r["Verdict"]:
                    st.error(f"🔴 **{r['Paire']}** | **{r['Verdict']}** | Tendance : {r['Tendance']} | RSI : {r['RSI']} | {r['Raison']}")
                else:
                    st.warning(f"🟡 **{r['Paire']}** | {r['Verdict']} | {r['Raison']}")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

# ============================================================
# PAGE 2 : ANALYSE DÉTAILLÉE
# ============================================================
elif page == "🔍 Analyse":
    st.markdown("# 🔍 Analyse Détaillée de la Paire")
    st.markdown("---")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        pair_name = st.selectbox("Paire d'actifs", list(PAIRS.keys()), index=0)
    with c2:
        tf_name = st.selectbox("Unité de temps", list(TIMEFRAMES.keys()), index=1)
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Analyser", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    tf_cfg = TIMEFRAMES[tf_name]
    df = fetch_data(PAIRS[pair_name], tf_cfg["period"], tf_cfg["interval"])
    res = analyze_market_v32(df, mode=sensibilite)

    if res:
        current_price = df.iloc[-1]['Close']
        st.markdown(f"""
        <div style='background-color: #1E222D; border-left: 8px solid {res["color"]}; padding: 20px; border-radius: 8px; margin-bottom: 20px;'>
            <h1 style='color: {res["color"]}; margin: 0;'>{res["verdict"]}</h1>
            <p style='color: #CCC; margin: 8px 0 0 0;'><b>{pair_name}</b> | {tf_name} | Prix direct : <b>{current_price:.5f}</b></p>
            <p style='color: #AAA; margin: 5px 0 0 0;'><b>Analyse :</b> {res["motif"]}</p>
        </div>
        """, unsafe_allow_html=True)

        if res["action"] in ["BUY", "SELL"]:
            st.markdown("### 📋 Ordre Recommandé pour MT5")
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Prix d'Entrée", f"{res['entry']:.5f}")
            x2.metric("Stop Loss", f"{res['sl']:.5f}")
            x3.metric("Take Profit", f"{res['tp']:.5f}")
            x4.metric("Ratio R:R", "1 : 1.8")

            if st.button("📝 Sauvegarder ce signal dans le Journal", use_container_width=True, type="primary"):
                ajouter_au_journal(
                    pair_name, res["verdict"], res["action"],
                    res["entry"], res["sl"], res["tp"], tf_name, res["motif"], res["score"]
                )
                st.success("Signal ajouté au journal avec succès !")
        else:
            st.info("💡 **Statut :** Aucune opportunité optimale actuellement. Si tu souhaites plus d'entrées, passe le mode en 'Dynamique' dans le menu latéral.")

        st.markdown("### 📊 Données Techniques")
        d1, d2, d3 = st.columns(3)
        d1.metric("Tendance EMA", res["trend"])
        d2.metric("RSI (14)", res["rsi"])
        d3.metric("ATR (Volatilité)", res["atr"])

# ============================================================
# PAGE 3 : GRAPHIQUE
# ============================================================
elif page == "📈 Graphique":
    st.markdown("# 📈 Graphique Interactif")
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
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=520)
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
    st.markdown("# 📓 Journal des Signaux Déclenchés")
    st.markdown("---")
    if len(st.session_state.journal) == 0:
        st.info("📭 Aucun signal pour l'instant. Lance un scan ou une analyse pour en ajouter.")
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
    st.markdown("# ⚙️ Guide d'utilisation")
    st.markdown("""
    ### Comment utiliser les 3 modes de sensibilité :
    - **Dynamique :** Idéal si tu veux plusieurs opportunités par jour (déclenchement dès 55% de confirmation).
    - **Modéré (Par défaut) :** Équilibre parfait entre nombre de signaux et fiabilité (déclenchement à 65%).
    - **Strict :** Pour ceux qui ne veulent que les configurations parfaites (déclenchement à 80%).
    """)

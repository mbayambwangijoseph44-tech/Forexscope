import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timezone

st.set_page_config(page_title="ForexScope Pro v4.0", page_icon="🎯", layout="wide")

# ============================================================
# JOURNAL
# ============================================================
if "journal" not in st.session_state:
    st.session_state.journal = []

def ajouter_signal(paire, verdict, action, entry, sl, tp, tf, motif, score):
    if action in ["BUY", "SELL"]:
        for s in st.session_state.journal[:5]:
            if s["Paire"] == paire and s["Verdict"] == verdict:
                return
        st.session_state.journal.insert(0, {
            "Date/Heure": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Paire": paire,
            "TF": tf,
            "Verdict": verdict,
            "Score": f"{score}%",
            "Entree": round(entry, 5) if entry else 0,
            "Stop Loss": round(sl, 5) if sl else 0,
            "Take Profit": round(tp, 5) if tp else 0,
            "Motif": motif,
            "Statut": "Actif"
        })

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

st.sidebar.markdown("### 🎛️ Sensibilite")
sensibilite = st.sidebar.selectbox(
    "Mode",
    ["Modere (Recommande)", "Dynamique (Plus de signaux)", "Strict (Haute precision)"],
    index=0
)

heure_utc = datetime.now(timezone.utc)
heure_actuelle = heure_utc.strftime("%H:%M:%S UTC")
heure_num = heure_utc.hour

if 7 <= heure_num < 16:
    session_txt = "Londres (Active)"
    session_icon = "🟢"
elif 12 <= heure_num < 20:
    session_txt = "New York (Active)"
    session_icon = "🟢"
else:
    session_txt = "Asie / Nuit (Bloquee)"
    session_icon = "🔴"

st.sidebar.markdown("---")
st.sidebar.caption(f"🕐 {heure_actuelle}")
st.sidebar.caption(f"Session : {session_icon} {session_txt}")
st.sidebar.caption(f"📓 Journal : {len(st.session_state.journal)} signaux")
st.sidebar.caption("v4.0 - Protections Completes")

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
# TELECHARGEMENT SECURISE
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(symbol, period, interval):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval, timeout=5)
        if df is None or df.empty or len(df) < 30:
            return None
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        return df
    except Exception:
        return None

# ============================================================
# MOTEUR D'ANALYSE v4.0 - TOUTES LES PROTECTIONS
# ============================================================
def analyze_market(df_ltf, df_htf, mode):
    if df_ltf is None or len(df_ltf) < 30:
        return None

    df = df_ltf.copy()

    # Indicateurs LTF
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # RSI
    diff = df['Close'].diff()
    gain = diff.clip(lower=0).rolling(14).mean()
    loss = (-diff.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift()))
    )
    df['ATR'] = tr.rolling(14).mean()

    # ADX
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr14 = df['ATR']
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-9))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(14).mean()

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']

    # Bougie cloturee N-2 (ZERO REPAINT)
    c = df.iloc[-2]
    prev = df.iloc[-3]
    atr_val = c['ATR'] if not np.isnan(c['ATR']) else (c['High'] - c['Low'])
    adx_val = c['ADX'] if not np.isnan(c['ADX']) else 25

    body = abs(c['Close'] - c['Open'])
    upper_wick = c['High'] - max(c['Close'], c['Open'])
    lower_wick = min(c['Close'], c['Open']) - c['Low']
    total_range = c['High'] - c['Low']

    # ========================================
    # FILTRE 1 : KILL-ZONE SESSION
    # ========================================
    now_h = datetime.now(timezone.utc).hour
    is_session_active = (7 <= now_h < 20)

    # ========================================
    # FILTRE 2 : MULTI-TIMEFRAME H1
    # ========================================
    htf_trend = "Neutre"
    if df_htf is not None and len(df_htf) > 50:
        htf_ema20 = df_htf['Close'].ewm(span=20, adjust=False).mean().iloc[-2]
        htf_ema50 = df_htf['Close'].ewm(span=50, adjust=False).mean().iloc[-2]
        if htf_ema20 > htf_ema50:
            htf_trend = "Haussiere"
        elif htf_ema20 < htf_ema50:
            htf_trend = "Baissiere"

    # ========================================
    # FILTRE 3 : VOLATILITE ATR MINIMUM
    # ========================================
    atr_pct = (atr_val / c['Close']) * 100 if c['Close'] > 0 else 0
    atr_suffisant = atr_pct > 0.02

    # Tendance LTF
    trend_up = c['EMA20'] > c['EMA50'] and c['Close'] > c['EMA50']
    trend_down = c['EMA20'] < c['EMA50'] and c['Close'] < c['EMA50']
    is_ranging = adx_val < 22

    # Zone de valeur
    in_buy_zone = c['Low'] <= c['EMA20'] * 1.001
    in_sell_zone = c['High'] >= c['EMA20'] * 0.999

    # ========================================
    # FILTRE 4 : REJET DURCI (1.8x le corps)
    # ========================================
    if body > 0:
        bullish_reject = (lower_wick >= 1.8 * body) or (c['Close'] > prev['High'] and c['Close'] > c['Open'] and body > 0.5 * total_range)
        bearish_reject = (upper_wick >= 1.8 * body) or (c['Close'] < prev['Low'] and c['Close'] < c['Open'] and body > 0.5 * total_range)
    else:
        bullish_reject = lower_wick > total_range * 0.6
        bearish_reject = upper_wick > total_range * 0.6

    # MACD confirmation
    macd_bull = c['Hist'] > prev['Hist']
    macd_bear = c['Hist'] < prev['Hist']

    # Seuil selon sensibilite
    if "Dynamique" in mode:
        seuil = 55
    elif "Strict" in mode:
        seuil = 80
    else:
        seuil = 65

    # ========================================
    # CALCUL DU SCORE PONDERE
    # ========================================
    score_buy = 0
    score_sell = 0
    motifs = []

    if trend_up:
        score_buy += 25
        motifs.append("Tendance LTF haussiere")
    if trend_down:
        score_sell += 25
        motifs.append("Tendance LTF baissiere")

    if htf_trend == "Haussiere":
        score_buy += 15
    elif htf_trend == "Baissiere":
        score_sell += 15

    if in_buy_zone:
        score_buy += 15
        motifs.append("Pullback zone EMA")
    if in_sell_zone:
        score_sell += 15
        motifs.append("Retest zone EMA")

    if bullish_reject:
        score_buy += 25
        motifs.append("Rejet acheteur fort")
    if bearish_reject:
        score_sell += 25
        motifs.append("Rejet vendeur fort")

    if 40 <= c['RSI'] <= 65:
        score_buy += 10
        score_sell += 10
    if macd_bull:
        score_buy += 10
    if macd_bear:
        score_sell += 10

    # ========================================
    # VERDICT FINAL AVEC TOUS LES FILTRES
    # ========================================
    verdict = "ATTENDRE"
    color = "#FFA500"
    action = "HOLD"
    final_score = max(score_buy, score_sell)
    entry = sl = tp = None

    if not is_session_active:
        motif_final = "🛑 KILL-ZONE : Session Asie/Nuit. Trades bloques pour proteger ton capital."
    elif not atr_suffisant:
        motif_final = "🛑 Volatilite trop faible (ATR insuffisant). Marche endormi."
    elif is_ranging:
        motif_final = "🟡 Marche en range (ADX < 22). Pas de direction claire."
    elif trend_up and htf_trend == "Baissiere":
        motif_final = "⚠️ Conflit MTF : LTF haussier mais H1 baissier. Trop risqué."
    elif trend_down and htf_trend == "Haussiere":
        motif_final = "⚠️ Conflit MTF : LTF baissier mais H1 haussier. Trop risqué."
    elif score_buy >= seuil and score_buy > score_sell and trend_up and htf_trend != "Baissiere":
        verdict = f"🟢 ACHAT ({score_buy}%)"
        color = "#00FF88"
        action = "BUY"
        final_score = score_buy
        motif_final = " | ".join(motifs)
        entry = df.iloc[-1]['Open']
        sl = round(c['Low'] - (0.4 * atr_val), 5)
        risk = max(entry - sl, 0.0005)
        tp = round(entry + (2.0 * risk), 5)
    elif score_sell >= seuil and score_sell > score_buy and trend_down and htf_trend != "Haussiere":
        verdict = f"🔴 VENTE ({score_sell}%)"
        color = "#FF3366"
        action = "SELL"
        final_score = score_sell
        motif_final = " | ".join(motifs)
        entry = df.iloc[-1]['Open']
        sl = round(c['High'] + (0.4 * atr_val), 5)
        risk = max(sl - entry, 0.0005)
        tp = round(entry - (2.0 * risk), 5)
    else:
        motif_final = f"Score insuffisant ({final_score}% / {seuil}% requis). Configuration incomplete."

    return {
        "verdict": verdict, "color": color, "action": action,
        "motif": motif_final, "entry": entry, "sl": sl, "tp": tp,
        "score": final_score, "rsi": round(c['RSI'], 1),
        "atr": round(atr_val, 5), "adx": round(adx_val, 1),
        "htf_trend": htf_trend,
        "trend": "Haussiere" if trend_up else ("Baissiere" if trend_down else "Range"),
        "price": df.iloc[-1]['Close'],
        "session": "Active" if is_session_active else "Bloquee"
    }

# ============================================================
# PAGE 1 : TABLEAU DE BORD
# ============================================================
if page == "📊 Tableau de bord":
    st.markdown("# 📊 Tableau de Bord")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        try:
            test = yf.Ticker("AUDUSD=X")
            h = test.history(period="1d", interval="1h", timeout=4)
            if not h.empty:
                st.success("🟢 Yahoo Finance : Connecte")
            else:
                st.error("🔴 Yahoo Finance : Pas de donnees")
        except Exception:
            st.error("🔴 Yahoo Finance : Erreur")
    with c2:
        st.info(f"🕐 {heure_actuelle}")
    with c3:
        if "🟢" in session_icon:
            st.success(f"Session : {session_txt}")
        else:
            st.warning(f"Session : {session_txt}")
    with c4:
        if st.button("🔄", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("### Scanner Multi-Paires (avec MTF H1)")
    tf_scan = st.selectbox("Unite de temps LTF", list(TIMEFRAMES.keys()), index=1)
    tf_cfg = TIMEFRAMES[tf_scan]

    if st.button("🚀 Lancer le Scan Complet", use_container_width=True, type="primary"):
        results = []
        bar = st.progress(0)
        for i, (name, sym) in enumerate(PAIRS.items()):
            df_ltf = fetch_data(sym, tf_cfg["period"], tf_cfg["interval"])
            df_htf = fetch_data(sym, "3mo", "1h")
            res = analyze_market(df_ltf, df_htf, sensibilite)
            if res:
                if res["action"] in ["BUY", "SELL"]:
                    ajouter_signal(name, res["verdict"], res["action"], res["entry"], res["sl"], res["tp"], tf_scan, res["motif"], res["score"])
                results.append({
                    "Paire": name,
                    "Verdict": res["verdict"],
                    "H1": res["htf_trend"],
                    "Tendance": res["trend"],
                    "RSI": res["rsi"],
                    "ADX": res["adx"],
                    "Raison": res["motif"]
                })
            bar.progress((i + 1) / len(PAIRS))

        if results:
            for r in results:
                if "ACHAT" in r["Verdict"]:
                    st.success(f"🟢 **{r['Paire']}** | {r['Verdict']} | H1: {r['H1']} | ADX: {r['ADX']} | {r['Raison']}")
                elif "VENTE" in r["Verdict"]:
                    st.error(f"🔴 **{r['Paire']}** | {r['Verdict']} | H1: {r['H1']} | ADX: {r['ADX']} | {r['Raison']}")
                else:
                    st.warning(f"🟡 **{r['Paire']}** | {r['Verdict']} | {r['Raison']}")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

# ============================================================
# PAGE 2 : ANALYSE DETAILLEE
# ============================================================
elif page == "🔍 Analyse":
    st.markdown("# 🔍 Analyse Detaillee Multi-Timeframe")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        p_name = st.selectbox("Paire", list(PAIRS.keys()), index=0)
    with col2:
        tf_name = st.selectbox("Unite de temps", list(TIMEFRAMES.keys()), index=1)

    tf_cfg = TIMEFRAMES[tf_name]

    with st.spinner("Analyse LTF + H1 en cours..."):
        df_ltf = fetch_data(PAIRS[p_name], tf_cfg["period"], tf_cfg["interval"])
        df_htf = fetch_data(PAIRS[p_name], "3mo", "1h")
        res = analyze_market(df_ltf, df_htf, sensibilite)

    if res:
        st.markdown(f"""
        <div style='background-color: #1E222D; border-left: 8px solid {res["color"]}; padding: 20px; border-radius: 8px; margin-bottom: 15px;'>
            <h1 style='color: {res["color"]}; margin: 0;'>{res["verdict"]}</h1>
            <p style='color: #CCC; margin: 8px 0 0 0;'><b>{p_name}</b> | {tf_name} | Prix : <b>{res['price']:.5f}</b> | Tendance H1 : <b>{res['htf_trend']}</b></p>
            <p style='color: #AAA; margin: 5px 0 0 0;'><b>Analyse :</b> {res["motif"]}</p>
        </div>
        """, unsafe_allow_html=True)

        if res["action"] in ["BUY", "SELL"]:
            st.markdown("### 📋 Ordre MT5")
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Prix Entree", f"{res['entry']:.5f}")
            x2.metric("Stop Loss", f"{res['sl']:.5f}")
            x3.metric("Take Profit", f"{res['tp']:.5f}")
            x4.metric("Ratio R:R", "1 : 2.0")

            if st.button("📝 Sauvegarder dans le Journal", use_container_width=True, type="primary"):
                ajouter_signal(p_name, res["verdict"], res["action"], res["entry"], res["sl"], res["tp"], tf_name, res["motif"], res["score"])
                st.success("Signal enregistre !")
        else:
            st.info("💡 Pas d'entree valide. Patience = rentabilite.")

        st.markdown("### 📊 Indicateurs")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Tendance H1", res["htf_trend"])
        d2.metric("Tendance LTF", res["trend"])
        d3.metric("RSI (14)", res["rsi"])
        d4.metric("ADX", res["adx"])
        d5.metric("ATR", res["atr"])
    else:
        st.warning("Donnees insuffisantes. Reessaye dans quelques secondes.")

# ============================================================
# PAGE 3 : GRAPHIQUE
# ============================================================
elif page == "📈 Graphique":
    st.markdown("# 📈 Graphique Interactif")
    st.markdown("---")
    p_name = st.selectbox("Paire", list(PAIRS.keys()), index=0)
    tf_name = st.selectbox("Unite de temps", list(TIMEFRAMES.keys()), index=1)
    tf_cfg = TIMEFRAMES[tf_name]

    df = fetch_data(PAIRS[p_name], tf_cfg["period"], tf_cfg["interval"])
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

        # RSI
        diff = df['Close'].diff()
        g = diff.clip(lower=0).rolling(14).mean()
        l = (-diff.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + g / (l + 1e-9)))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='#BB86FC', width=1.5), name="RSI"))
        fig2.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
        fig2.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
        fig2.update_layout(template="plotly_dark", title="RSI (14)", height=180, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# PAGE 4 : RISQUE
# ============================================================
elif page == "⚠️ Risque":
    st.markdown("# ⚠️ Calculateur de Lot MT5")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    cap = c1.number_input("Capital ($)", value=1000.0, step=100.0)
    risk = c2.number_input("Risque max (%)", value=1.0, max_value=5.0, step=0.5)
    sl_p = c3.number_input("Stop Loss (pips)", value=20.0, step=1.0)

    montant = cap * (risk / 100)
    lot = montant / (sl_p * 10.0)

    r1, r2 = st.columns(2)
    r1.metric("Montant Risque", f"${montant:.2f}")
    r2.metric("Lot Recommande", f"{lot:.2f} lots")
    st.info("Regle d'or : Jamais plus de 1-2% de risque par trade.")

# ============================================================
# PAGE 5 : JOURNAL
# ============================================================
elif page == "📓 Journal":
    st.markdown("# 📓 Journal des Signaux")
    st.markdown("---")
    if len(st.session_state.journal) == 0:
        st.info("📭 Aucun signal enregistre.")
    else:
        total = len(st.session_state.journal)
        buys = len([s for s in st.session_state.journal if "ACHAT" in s["Verdict"]])
        sells = len([s for s in st.session_state.journal if "VENTE" in s["Verdict"]])

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Total", total)
        sc2.metric("Achats", buys)
        sc3.metric("Ventes", sells)

        st.markdown("---")
        df_j = pd.DataFrame(st.session_state.journal)
        st.dataframe(df_j, use_container_width=True, hide_index=True)

        st.markdown("---")
        col_exp, col_clr = st.columns(2)
        with col_exp:
            csv = df_j.to_csv(index=False).encode('utf-8')
            st.downloa

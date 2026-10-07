import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timezone

st.set_page_config(page_title="ForexScope Pro v4.1", page_icon="🎯", layout="wide")

# ============================================================
# JOURNAL & SESSION
# ============================================================
if "journal" not in st.session_state:
    st.session_state.journal = []

if "sensibilite" not in st.session_state:
    st.session_state.sensibilite = "Strict (Haute precision)"

def ajouter_signal(paire, verdict, action, entry, sl, tp, tf, motif, score, heure_signal):
    if action in ["BUY", "SELL"]:
        for s in st.session_state.journal[:5]:
            if s["Paire"] == paire and s["Verdict"] == verdict:
                return
        st.session_state.journal.insert(0, {
            "Heure Signal": heure_signal,
            "Heure Sauvegarde": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "Paire": paire,
            "TF": tf,
            "Verdict": verdict,
            "Score": str(score) + "%",
            "Entree": round(entry, 5) if entry else 0,
            "Stop Loss": round(sl, 5) if sl else 0,
            "Take Profit": round(tp, 5) if tp else 0,
            "Motif": motif,
            "Statut": "Actif"
        })

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

st.sidebar.markdown("### 🎛️ Sensibilite")
sensibilite = st.sidebar.selectbox(
    "Mode d'analyse",
    ["Strict (Haute precision)", "Modere (Recommande)", "Dynamique (Plus de signaux)"],
    key="sensibilite"
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
st.sidebar.caption("🕐 " + heure_actuelle)
st.sidebar.caption("Session : " + session_icon + " " + session_txt)
st.sidebar.caption("📓 Journal : " + str(len(st.session_state.journal)) + " signaux")
st.sidebar.caption("v4.1 - Protections Completes")

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
    "M15 (15 min)": {"interval": "15m", "period": "1mo"},
    "M5 (5 min)": {"interval": "5m", "period": "5d"},
    "M30 (30 min)": {"interval": "30m", "period": "1mo"},
    "H1 (1 heure)": {"interval": "1h", "period": "3mo"},
}

# ============================================================
# TELECHARGEMENT SECURISE
# ============================================================
@st.cache_data(ttl=45, show_spinner=False)
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
# MOTEUR D'ANALYSE v4.1
# ============================================================
def analyze_market(df_ltf, df_htf, mode):
    if df_ltf is None or len(df_ltf) < 35:
        return None

    df = df_ltf.copy()

    # Moyennes Mobiles
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

    total_range = c['High'] - c['Low']
    body = abs(c['Close'] - c['Open'])
    lower_wick = min(c['Close'], c['Open']) - c['Low']
    upper_wick = c['High'] - max(c['Close'], c['Open'])

    signal_time = str(df.index[-2])

    # 1. Filtre Horaire
    now_h = datetime.now(timezone.utc).hour
    is_session_active = (7 <= now_h < 20)

    # 2. Filtre Tendance H1
    htf_trend = "Neutre"
    if df_htf is not None and len(df_htf) > 50:
        htf_e20 = df_htf['Close'].ewm(span=20, adjust=False).mean().iloc[-2]
        htf_e50 = df_htf['Close'].ewm(span=50, adjust=False).mean().iloc[-2]
        if htf_e20 > htf_e50:
            htf_trend = "Haussiere"
        elif htf_e20 < htf_e50:
            htf_trend = "Baissiere"

    # 3. Filtre Pente EMA50
    if len(df) >= 7:
        ema50_slope = df['EMA50'].iloc[-2] - df['EMA50'].iloc[-7]
    else:
        ema50_slope = 0
    is_trending_up = ema50_slope > (0.05 * atr_val)
    is_trending_down = ema50_slope < -(0.05 * atr_val)

    # 4. Volatilite ATR
    atr_pct = (atr_val / c['Close']) * 100 if c['Close'] > 0 else 0
    atr_ok = atr_pct > 0.02

    # 5. Rejet Pinbar Strict
    if total_range > 0:
        is_bull_pinbar = (lower_wick >= 0.6 * total_range) and (body <= 0.3 * total_range)
        is_bear_pinbar = (upper_wick >= 0.6 * total_range) and (body <= 0.3 * total_range)
    else:
        is_bull_pinbar = False
        is_bear_pinbar = False

    is_bull_engulf = (c['Close'] > prev['High']) and (c['Close'] > c['Open'])
    is_bear_engulf = (c['Close'] < prev['Low']) and (c['Close'] < c['Open'])

    trend_up = c['EMA20'] > c['EMA50'] and c['Close'] > c['EMA50']
    trend_down = c['EMA20'] < c['EMA50'] and c['Close'] < c['EMA50']
    is_ranging = adx_val < 22

    in_buy_zone = c['Low'] <= c['EMA20'] * 1.001
    in_sell_zone = c['High'] >= c['EMA20'] * 0.999

    macd_bull = c['Hist'] > prev['Hist']
    macd_bear = c['Hist'] < prev['Hist']

    seuil = 55 if "Dynamique" in mode else (75 if "Strict" in mode else 65)

    score_buy = 0
    score_sell = 0
    motifs = []

    if trend_up and is_trending_up:
        score_buy += 30
        motifs.append("Tendance haussiere (EMA inclinee)")
    if trend_down and is_trending_down:
        score_sell += 30
        motifs.append("Tendance baissiere (EMA inclinee)")

    if htf_trend == "Haussiere":
        score_buy += 15
    elif htf_trend == "Baissiere":
        score_sell += 15

    if in_buy_zone:
        score_buy += 15
        motifs.append("Pullback EMA")
    if in_sell_zone:
        score_sell += 15
        motifs.append("Retest EMA")

    if is_bull_pinbar:
        score_buy += 30
        motifs.append("Pinbar rejet haussier")
    elif is_bull_engulf:
        score_buy += 20
        motifs.append("Avalement haussier")

    if is_bear_pinbar:
        score_sell += 30
        motifs.append("Pinbar rejet baissier")
    elif is_bear_engulf:
        score_sell += 20
        motifs.append("Avalement baissier")

    if 40 <= c['RSI'] <= 65:
        score_buy += 10
        score_sell += 10
    if macd_bull:
        score_buy += 10
    if macd_bear:
        score_sell += 10

    verdict = "ATTENDRE"
    color = "#FFA500"
    action = "HOLD"
    final_score = max(score_buy, score_sell)
    entry = sl = tp = None

    if not is_session_active:
        motif_final = "KILL-ZONE : Session Asie/Nuit. Trades bloques."
    elif not atr_ok:
        motif_final = "Volatilite trop faible (ATR). Marche endormi."
    elif is_ranging:
        motif_final = "Marche en range (ADX < 22). Pas de direction."
    elif not is_trending_up and not is_trending_down:
        motif_final = "EMA50 plate : marche plat. Attendre impulsion."
    elif trend_up and htf_trend == "Baissiere":
        motif_final = "Conflit MTF : LTF haussier mais H1 baissier. Trop risque."
    elif trend_down and htf_trend == "Haussiere":
        motif_final = "Conflit MTF : LTF baissier mais H1 haussier. Trop risque."
    elif score_buy >= seuil and score_buy > score_sell and htf_trend != "Baissiere":
        verdict = "ACHAT (" + str(score_buy) + "%)"
        color = "#00FF88"
        action = "BUY"
        final_score = score_buy
        motif_final = " | ".join(motifs)
        entry = df.iloc[-1]['Open']
        recent_low = df['Low'].iloc[-7:-2].min()
        sl = round(min(recent_low, entry - (1.5 * atr_val)), 5)
        risk = entry - sl
        if risk < 0.00180:
            sl = round(entry - 0.00180, 5)
            risk = 0.00180
        tp = round(entry + (1.8 * risk), 5)
    elif score_sell >= seuil and score_sell > score_buy and htf_trend != "Haussiere":
        verdict = "VENTE (" + str(score_sell) + "%)"
        color = "#FF3366"
        action = "SELL"
        final_score = score_sell
        motif_final = " | ".join(motifs)
        entry = df.iloc[-1]['Open']
        recent_high = df['High'].iloc[-7:-2].max()
        sl = round(max(recent_high, entry + (1.5 * atr_val)), 5)
        risk = sl - entry
        if risk < 0.00180:
            sl = round(entry + 0.00180, 5)
            risk = 0.00180
        tp = round(entry - (1.8 * risk), 5)
    else:
        motif_final = "Score insuffisant (" + str(final_score) + "% / " + str(seuil) + "% requis)."

    return {
        "verdict": verdict,
        "color": color,
        "action": action,
        "motif": motif_final,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "score": final_score,
        "rsi": round(c['RSI'], 1),
        "atr": round(atr_val, 5),
        "adx": round(adx_val, 1),
        "htf_trend": htf_trend,
        "trend": "Haussiere" if trend_up else ("Baissiere" if trend_down else "Range"),
        "price": df.iloc[-1]['Close'],
        "signal_time": signal_time
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
        st.info("🕐 " + heure_actuelle)
    with c3:
        if session_icon == "🟢":
            st.success("Session : " + session_txt)
        else:
            st.warning("Session : " + session_txt)
    with c4:
        if st.button("🔄", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("### Scanner Multi-Paires")
    tf_scan = st.selectbox("Unite de temps LTF", list(TIMEFRAMES.keys()), index=0)
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
                    ajouter_signal(
                        name, res["verdict"], res["action"],
                        res["entry"], res["sl"], res["tp"],
                        tf_scan, res["motif"], res["score"],
                        res["signal_time"]
                    )
                results.append({
                    "Paire": name,
                    "Verdict": res["verdict"],
                    "H1": res["htf_trend"],
                    "Tendance": res["trend"],
                    "RSI": res["rsi"],
                    "ADX": res["adx"],
                    "Heure Signal": res["signal_time"],
                    "Raison": res["motif"]
                })
            bar.progress((i + 1) / len(PAIRS))

        if results:
            for r in results:
                if "ACHAT" in r["Verdict"]:
                    st.success("🟢 **" + r["Paire"] + "** | " + r["Verdict"] + " | H1: " + r["H1"] + " | " + r["Raison"])
                elif "VENTE" in r["Verdict"]:
                    st.error("🔴 **" + r["Paire"] + "** | " + r["Verdict"] + " | H1: " + r["H1"] + " | " + r["Raison"])
                else:
                    st.warning("🟡 **" + r["Paire"] + "** | " + r["Verdict"] + " | " + r["Raison"])
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        else:
            st.warning("Aucune donnee disponible. Reessaye dans quelques secondes.")

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
        tf_name = st.selectbox("Unite de temps", list(TIMEFRAMES.keys()), index=0)

    tf_cfg = TIMEFRAMES[tf_name]

    with st.spinner("Analyse LTF + H1 en cours..."):
        df_ltf = fetch_data(PAIRS[p_name], tf_cfg["period"], tf_cfg["interval"])
        df_htf = fetch_data(PAIRS[p_name], "3mo", "1h")
        res = analyze_market(df_ltf, df_htf, sensibilite)

    if res:
        st.markdown(
            "<div style='background-color: #1E222D; border-left: 8px solid "
            + res["color"]
            + "; padding: 20px; border-radius: 8px; margin-bottom: 15px;'>"
            + "<h1 style='color: " + res["color"] + "; margin: 0;'>" + res["verdict"] + "</h1>"
            + "<p style='color: #CCC; margin: 8px 0 0 0;'><b>" + p_name + "</b> | " + tf_name
            + " | Prix : <b>" + str(round(res["price"], 5)) + "</b>"
            + " | Tendance H1 : <b>" + res["htf_trend"] + "</b></p>"
            + "<p style='color: #AAA; margin: 5px 0 0 0;'><b>Analyse :</b> " + res["motif"] + "</p>"
            + "<p style='color: #666; font-size: 12px; margin: 5px 0 0 0;'>"
            + "Signal genere a : " + res["signal_time"] + "</p>"
            + "</div>",
            unsafe_allow_html=True
        )

        if res["action"] in ["BUY", "SELL"]:
            st.markdown("### 📋 Ordre MT5 Recommande")
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Prix Entree", str(round(res["entry"], 5)))
            x2.metric("Stop Loss", str(round(res["sl"], 5)))
            x3.metric("Take Profit", str(round(res["tp"], 5)))
            x4.metric("Ratio R:R", "1 : 1.8")

            if st.button("📝 Sauvegarder dans le Journal", use_container_width=True, type="primary"):
                ajouter_signal(
                    p_name, res["verdict"], res["action"],
                    res["entry"], res["sl"], res["tp"],
                    tf_name, res["motif"], res["score"],
                    res["signal_time"]
                )
                st.success("Signal enregistre !")
        else:
            st.info("Pas d'entree valide. Patience = rentabilite.")

        st.markdown("### 📊 Indicateurs")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Tendance H1", res["htf_trend"])
        d2.metric("Tendance LTF", res["trend"])
        d3.metric("RSI (14)", str(res["rsi"]))
        d4.metric("ADX", str(res["adx"]))
        d5.metric("ATR", str(res["atr"]))
    else:
        st.warning("Donnees insuffisantes. Reessaye dans quelques secondes.")

# ============================================================
# PAGE 3 : GRAPHIQUE
# ============================================================
elif page == "📈 Graphique":
    st.markdown("# 📈 Graphique Interactif")
    st.markdown("---")
    p_name = st.selectbox("Paire", list(PAIRS.keys()), index=0)
    tf_name = st.selectbox("Unite de temps", list(TIMEFRAMES.keys()), index=0)
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
    else:
        st.warning("Donnees indisponibles pour ce graphique.")

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
    r1.metric("Montant Risque", "$" + str(round(montant, 2)))
    r2.metric("Lot Recommande", str(round(lot, 2)) + " lots")
    st.info("Regle d'or : Jamais plus de 1-2% de risque par trade.")

# ============================================================
# PAGE 5 : JOURNAL
# ============================================================
elif page == "📓 Journal":
    st.markdown("# 📓 Journal des Signaux")
    st.markdown("---")
    if len(st.session_state.journal) == 0:
        st.info("Aucun signal enregistre.")
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
            st.download_button(
                "📥 Exporter CSV",
                csv,
                "forexscope_journal_" + datetime.now().strftime("%Y%m%d") + ".csv",
                "text/csv",
                use_container_width=True
            )
        with col_clr:
            if st.button("🗑️ Vider le journal", use_container_width=True):
                st.session_state.journal = []
                st.rerun()

# ============================================================
# PAGE 6 : PARAMETRES
# ============================================================
elif page == "⚙️ Parametres":
    st.markdown("# ⚙️ Parametres et Protections Actives")
    st.markdown("---")

    st.markdown("### 📡 Source de donnees")
    try:
        test = yf.Ticker("AUDUSD=X")
        h = test.history(period="1d", interval="1h", timeout=4)
        if not h.empty:
            st.success("🟢 Yahoo Finance : Connecte et operationnel")
        else:
            st.error("🔴 Yahoo Finance : Pas de donnees")
    except Exception:
        st.error("🔴 Yahoo Finance : Erreur de connexion")
    st.caption("Deriv WebSocket sera reactive sur VPS.")

    st.markdown("### 🛡️ Protections Actives")
    st.markdown("""
    | Filtre | Description | Statut |
    |--------|-------------|--------|
    | Kill-Zone Session | Bloque trades 20h-07h UTC | Actif |
    | Multi-Timeframe H1 | Interdit trader contre H1 | Actif |
    | Pente EMA50 | Bloque si EMA plate (range) | Actif |
    | Rejet Pinbar Strict | Meche >= 60% de la bougie | Actif |
    | ATR Minimum | Bloque si volatilite trop faible | Actif |
    | ADX Range | Bloque si ADX < 22 | Actif |
    | SL Structurel | Min 18 pips, base sur swing | Actif |
    | Zero Repaint | Analyse sur bougie N-2 cloturee | Actif |
    | MACD Confirmation | Bonus score si MACD confirme | Actif |
    """)

    st.markdown("### 🎛️ Modes de Sensibilite")
    st.markdown("""
    - **Strict (75%)** : Peu de signaux, haute fiabilite. Recommande.
    - **Modere (65%)** : Equilibre entre quantite et qualite.
    - **Dynamique (55%)** : Plus de signaux, ideal pour tester.
    """)

    st.markdown("### ⚠️ Avertissement")
    st.warning("ForexScope est un outil d'aide a la decision. Le trading comporte des risques de perte. Teste en DEMO avant le reel.")
    st.caption("ForexScope v4.1 | Protections Completes")
    
        

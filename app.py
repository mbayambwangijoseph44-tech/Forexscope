import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timezone

st.set_page_config(page_title="ForexScope Pro v4.1", page_icon="🎯", layout="wide")

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
# NAVIGATION
# ============================================================
st.sidebar.markdown("# 🎯 FOREXSCOPE")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Tableau de bord", "🔍 Analyse", "📈 Graphique", "⚠️ Risque", "📓 Journal", "⚙️ Parametres"]
)
st.sidebar.markdown("---")

st.sidebar.markdown("### 🎛️ Configuration")
sensibilite = st.sidebar.selectbox(
    "Sensibilité",
    ["Modéré (Recommandé)", "Dynamique (Plus de signaux)", "Strict (Haute précision)"],
    index=0
)

# Horaires
heure_utc = datetime.now(timezone.utc)
heure_actuelle = heure_utc.strftime("%H:%M:%S UTC")
heure_num = heure_utc.hour

if 7 <= heure_num < 16:
    session_txt, session_icon = "Londres (Active)", "🟢"
elif 12 <= heure_num < 20:
    session_txt, session_icon = "New York (Active)", "🟢"
else:
    session_txt, session_icon = "Asie / Nuit (Bloquée)", "🔴"

st.sidebar.markdown("---")
st.sidebar.caption(f"🕐 {heure_actuelle}")
st.sidebar.caption(f"Session : {session_icon} {session_txt}")
st.sidebar.caption(f"v4.1 - Stop Loss & Rejets Corrigés")

# ============================================================
# CONFIGURATION ACTIFS
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
# MOTEUR D'ANALYSE RECORRIGÉ (v4.1)
# ============================================================
def analyze_market_v41(df_ltf, df_htf, mode):
    if df_ltf is None or len(df_ltf) < 35:
        return None

    df = df_ltf.copy()

    # Indicateurs
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

    # Analyse sur bougie clôturée N-2 (Anti-Repaint)
    c = df.iloc[-2]
    prev = df.iloc[-3]
    atr_val = c['ATR'] if not np.isnan(c['ATR']) else (c['High'] - c['Low'])

    total_range = c['High'] - c['Low']
    body = abs(c['Close'] - c['Open'])
    lower_wick = min(c['Close'], c['Open']) - c['Low']
    upper_wick = c['High'] - max(c['Close'], c['Open'])

    # --- 1. FILTRE HORAIRE ---
    now_h = datetime.now(timezone.utc).hour
    is_session_active = (7 <= now_h < 20)

    # --- 2. FILTRE TENDANCE SUPÉRIEURE (H1) ---
    htf_trend = "Neutre"
    if df_htf is not None and len(df_htf) > 50:
        htf_ema20 = df_htf['Close'].ewm(span=20, adjust=False).mean().iloc[-2]
        htf_ema50 = df_htf['Close'].ewm(span=50, adjust=False).mean().iloc[-2]
        if htf_ema20 > htf_ema50:
            htf_trend = "Haussiere"
        elif htf_ema20 < htf_ema50:
            htf_trend = "Baissiere"

    # --- 3. FILTRE PENTE EMA 50 (ANTI-RANGE) ---
    # Calcul de la pente sur les 5 dernières bougies
    ema50_slope = df['EMA50'].iloc[-2] - df['EMA50'].iloc[-7]
    is_trending_up = ema50_slope > (0.05 * atr_val)
    is_trending_down = ema50_slope < -(0.05 * atr_val)

    # --- 4. DÉFINITION STRICTE DU REJET (Pinbar Réel) ---
    # La mèche doit représenter au moins 60% de la bougie totale, et le corps < 30%
    is_bullish_pinbar = (lower_wick >= 0.6 * total_range) and (body <= 0.3 * total_range) and (total_range > 0)
    is_bearish_pinbar = (upper_wick >= 0.6 * total_range) and (body <= 0.3 * total_range) and (total_range > 0)

    # Rejet secondaire (Avalement Fort)
    is_bullish_engulfing = (c['Close'] > prev['High']) and (c['Close'] > c['Open'])
    is_bearish_engulfing = (c['Close'] < prev['Low']) and (c['Close'] < c['Open'])

    # Zone de retest
    in_buy_zone = c['Low'] <= c['EMA20'] * 1.001
    in_sell_zone = c['High'] >= c['EMA20'] * 0.999

    # Configuration Seuil
    seuil = 55 if "Dynamique" in mode else (75 if "Strict" in mode else 65)

    # Analyse Poids
    score_buy = 0
    score_sell = 0
    motifs = []

    if c['EMA20'] > c['EMA50'] and is_trending_up:
        score_buy += 30
        motifs.append("Tendance haussiere saine (EMA inclinée)")
    if c['EMA20'] < c['EMA50'] and is_trending_down:
        score_sell += 30
        motifs.append("Tendance baissiere saine (EMA inclinée)")

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

    if is_bullish_pinbar:
        score_buy += 30
        motifs.append("Pinbar de Rejet haussier")
    elif is_bullish_engulfing:
        score_buy += 20
        motifs.append("Avalement haussier")

    if is_bearish_pinbar:
        score_sell += 30
        motifs.append("Pinbar de Rejet baissier")
    elif is_bearish_engulfing:
        score_sell += 20
        motifs.append("Avalement baissier")

    # Décision
    verdict = "ATTENDRE"
    color = "#FFA500"
    action = "HOLD"
    final_score = max(score_buy, score_sell)
    entry = sl = tp = None

    if not is_session_active:
        motif_final = "🛑 KILL-ZONE : Marché Asiatique peu liquide. Risque élevé."
    elif not is_trending_up and not is_trending_down:
        motif_final = "🟡 EMA50 plate : Marché plat (Range). Attendre une impulsion."
    elif score_buy >= seuil and score_buy > score_sell and htf_trend != "Baissiere":
        verdict = f"🟢 ACHAT ({score_buy}%)"
        color = "#00FF88"
        action = "BUY"
        motif_final = " | ".join(motifs)
        entry = df.iloc[-1]['Open']
        
        # --- CALCUL SL STRUCTUREL ---
        # Plus bas des 5 dernières bougies - un petit filtre de sécurité
        recent_low = df['Low'].iloc[-7:-2].min()
        sl = round(min(recent_low, entry - (1.5 * atr_val)), 5)
        risk = entry - sl
        tp = round(entry + (1.8 * risk), 5)
        
    elif score_sell >= seuil and score_sell > score_buy and htf_trend != "Haussiere":
        verdict = f"🔴 VENTE ({score_sell}%)"
        color = "#FF3366"
        action = "SELL"
        motif_final = " | ".join(motifs)
        entry = df.iloc[-1]['Open']
        
        # --- CALCUL SL STRUCTUREL ---
        # Plus haut des 5 dernières bougies + un petit filtre de sécurité
        recent_high = df['High'].iloc[-7:-2].max()
        sl = round(max(recent_high, entry + (1.5 * atr_val)), 5)
        risk = sl - entry
        tp = round(entry - (1.8 * risk), 5)
    else:
        motif_final = f"Score insuffisant ({final_score}% / {seuil}% requis)."

    # Empêcher les SL absurdes de moins de 15 pips (sécurité absolue)
    if entry and sl:
        pips_dist = abs(entry - sl) if "JPY" not in df_ltf.columns else abs(entry - sl) / 100
        # Marge minimale de sécurité (18 pips pour paires standards, 180 points pour JPY)
        min_pips = 0.00180 if "JPY=X" not in df_ltf.columns else 0.18
        if action == "BUY" and (entry - sl) < min_pips:
            sl = round(entry - min_pips, 5)
            tp = round(entry + (1.8 * min_pips), 5)
        elif action == "SELL" and (sl - entry) < min_pips:
            sl = round(entry + min_pips, 5)
            tp = round(entry - (1.8 * min_pips), 5)

    return {
        "verdict": verdict, "color": color, "action": action,
        "motif": motif_final, "entry": entry, "sl": sl, "tp": tp,
        "score": final_score, "rsi": round(c['RSI'], 1),
        "atr": round(atr_val, 5), "trend_htf": htf_trend,
        "price": df.iloc[-1]['Close']
    }

# ============================================================
# PAGE INTERFACE
# ============================================================
if page == "📊 Tableau de bord":
    st.markdown("# 📊 Tableau de Bord")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.success("🟢 Flux Connecté (Yahoo)")
    c2.info(f"🕐 UTC : {heure_actuelle}")
    c3.warning(f"Session : {session_txt}")

    st.markdown("---")
    tf_scan = st.selectbox("Unité de temps", list(TIMEFRAMES.keys()), index=0)
    tf_cfg = TIMEFRAMES[tf_scan]

    if st.button("🚀 Scanner les 8 paires", use_container_width=True, type="primary"):
        results = []
        bar = st.progress(0)
        for i, (name, sym) in enumerate(PAIRS.items()):
            df_ltf = fetch_data(sym, tf_cfg["period"], tf_cfg["interval"])
            df_htf = fetch_data(sym, "3mo", "1h")
            res = analyze_market_v41(df_ltf, df_htf, sensibilite)
            if res:
                if res["action"] in ["BUY", "SELL"]:
                    ajouter_signal(name, res["verdict"], res["action"], res["entry"], res["sl"], res["tp"], tf_scan, res["motif"], res["score"])
                results.append({
                    "Paire": name, "Verdict": res["verdict"], "H1": res["trend_htf"], "Prix": f"{res['price']:.5f}", "Analyse": res["motif"]
                })
            bar.progress((i + 1) / len(PAIRS))

        if results:
            for r in results:
                if "ACHAT" in r["Verdict"]:
                    st.success(f"🟢 **{r['Paire']}** | {r['Verdict']} | H1: {r['H1']} | {r['Analyse']}")
                elif "VENTE" in r["Verdict"]:
                    st.error(f"🔴 **{r['Paire']}** | {r['Verdict']} | H1: {r['H1']} | {r['Analyse']}")
                else:
                    st.warning(f"🟡 **{r['Paire']}** | {r['Verdict']} | {r['Analyse']}")

elif page == "🔍 Analyse":
    st.markdown("# 🔍 Analyse de Paire")
    st.markdown("---")

    col1, col2 = st.columns(2)
    p_name = col1.selectbox("Paire", list(PAIRS.keys()), index=0)
    tf_name = col2.selectbox("Unité de temps", list(TIMEFRAMES.keys()), index=0)

    tf_cfg = TIMEFRAMES[tf_name]
    df_ltf = fetch_data(PAIRS[p_name], tf_cfg["period"], tf_cfg["interval"])
    df_htf = fetch_data(PAIRS[p_name], "3mo", "1h")
    res = analyze_market_v41(df_ltf, df_htf, sensibilite)

    if res:
        st.markdown(f"""
        <div style='background-color: #1E222D; border-left: 8px solid {res["color"]}; padding: 20px; border-radius: 8px;'>
            <h1 style='color: {res["color"]}; margin: 0;'>{res["verdict"]}</h1>
            <p style='color: #CCC; margin: 8px 0 0 0;'><b>{p_name}</b> | {tf_name} | Prix actuel : <b>{res['price']:.5f}</b></p>
            <p style='color: #AAA; margin: 5px 0 0 0;'><b>Filtres :</b> {res["motif"]}</p>
        </div>
        """, unsafe_allow_html=True)

        if res["action"] in ["BUY", "SELL"]:
            st.markdown("### 📋 Paramètres MT5 Recommandés")
            x1, x2, x3 = st.columns(3)
            x1.metric("Prix Entrée", f"{res['entry']:.5f}")
            x2.metric("Stop Loss (SÉCURISÉ)", f"{res['sl']:.5f}")
            x3.metric("Take Profit (R:R 1:1.8)", f"{res['tp']:.5f}")

            if st.button("Sauvegarder dans le Journal"):
                ajouter_signal(p_name, res["verdict"], res["action"], res["entry"], res["sl"], res["tp"], tf_name, res["motif"], res["score"])
                st.success("Sauvegardé !")

elif page == "📈 Graphique":
    st.markdown("# 📈 Graphique Chandelier")
    st.markdown("---")
    p_name = st.selectbox("Paire", list(PAIRS.keys()))
    tf_name = st.selectbox("Unité de temps", list(TIMEFRAMES.keys()))
    df = fetch_data(PAIRS[p_name], TIMEFRAMES[tf_name]["period"], TIMEFRAMES[tf_name]["interval"])

    if df is not None and len(df) > 10:
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

elif page == "⚠️ Risque":
    st.markdown("# ⚠️ Risque & Lot")
    cap = st.number_input("Capital ($)", value=1000.0)
    risk = st.number_input("Risque (%)", value=1.0)
    sl_p = st.number_input("Stop Loss (pips)", value=20.0)
    st.metric("Taille du Lot Recommandée", f"{round((cap * (risk/100)) / (sl_p * 10), 2)} lots")

elif page == "📓 Journal":
    st.markdown("# 📓 Journal des Signaux")
    if len(st.session_state.journal) == 0:
        st.info("Aucun signal enregistré.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.journal), use_container_width=True)
        if st.button("Effacer tout"):
            st.session_state.journal = []
            st.rerun()

elif page == "⚙️ Parametres":
    st.markdown("# ⚙️ Paramètres de protection")
    st.write("Filtre d'inclinaison d'EMA50 actif : Bloque les phases de range.")
    st.write("Filtre d'écartement minimum de Stop Loss : Minimum 18 pips pour laisser respirer le trade.")

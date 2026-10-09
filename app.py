import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timezone, timedelta
import os

st.set_page_config(page_title="ForexScope Pro v5.2 - Confluence Matrix", page_icon="🎯", layout="wide")

# ============================================================
# JOURNAL PERSISTANT & COOLDOWN DYNAMIQUE
# ============================================================
JOURNAL_FILE = "forexscope_journal.csv"

TF_SECONDS = {
    "M5 (5 min)": 300,
    "M15 (15 min)": 900,
    "M30 (30 min)": 1800,
    "H1 (1 heure)": 3600
}

def charger_journal():
    if os.path.exists(JOURNAL_FILE):
        try:
            df = pd.read_csv(JOURNAL_FILE)
            return df.to_dict("records")
        except Exception:
            return []
    return []

def sauvegarder_journal(journal):
    try:
        df = pd.DataFrame(journal)
        df.to_csv(JOURNAL_FILE, index=False)
    except Exception:
        pass

if "journal" not in st.session_state:
    st.session_state.journal = charger_journal()

if "sensibilite" not in st.session_state:
    st.session_state.sensibilite = "Strict (Haute precision)"

if "signaux_verrouilles" not in st.session_state:
    st.session_state.signaux_verrouilles = {}

def ajouter_signal(paire, verdict, action, entry, sl, tp, tf, motif, score, confluence_grade, heure_signal):
    if action not in ["BUY", "SELL"]:
        return False

    duree_bougie = TF_SECONDS.get(tf, 900)
    cooldown_requis = duree_bougie * 3

    now = datetime.now(timezone.utc)
    cle_verrou = paire + "_" + tf

    if cle_verrou in st.session_state.signaux_verrouilles:
        dernier_temps = st.session_state.signaux_verrouilles[cle_verrou]
        temps_ecoule = (now - dernier_temps).total_seconds()
        if temps_ecoule < cooldown_requis:
            return False

    for s in st.session_state.journal[:5]:
        if s.get("Paire") == paire and s.get("Verdict") == verdict and s.get("TF") == tf:
            return False

    st.session_state.journal.insert(0, {
        "Heure Signal": heure_signal,
        "Heure Sauvegarde": now.strftime("%Y-%m-%d %H:%M UTC"),
        "Paire": paire,
        "TF": tf,
        "Verdict": verdict,
        "Confluence": confluence_grade,
        "Score": str(score) + "%",
        "Entree": round(entry, 5) if entry else 0,
        "Stop Loss": round(sl, 5) if sl else 0,
        "Take Profit": round(tp, 5) if tp else 0,
        "Motif": motif,
        "Statut": "Actif"
    })

    st.session_state.signaux_verrouilles[cle_verrou] = now
    sauvegarder_journal(st.session_state.journal)
    return True

# ============================================================
# NAVIGATION & BARRE LATERALE
# ============================================================
st.sidebar.markdown("# 🎯 FOREXSCOPE Pro")
st.sidebar.caption("Système Décisionnel Multi-Confluence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Tableau de bord", "🔍 Analyse Confluence", "📈 Graphique", "⚠️ Risque", "📓 Journal", "⚙️ Parametres"]
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
    session_txt, session_icon = "Londres (Active)", "🟢"
elif 12 <= heure_num < 20:
    session_txt, session_icon = "New York (Active)", "🟢"
else:
    session_txt, session_icon = "Asie / Nuit (Bloquee)", "🔴"

st.sidebar.markdown("---")
st.sidebar.caption("🕐 " + heure_actuelle)
st.sidebar.caption("Session : " + session_icon + " " + session_txt)
st.sidebar.caption("📓 Journal : " + str(len(st.session_state.journal)) + " signaux")
st.sidebar.caption("v5.2 - Matrice de Confluence")

PAIRS = {
    "AUD/USD": "AUDUSD=X", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X", "USD/CAD": "USDCAD=X", "EUR/GBP": "EURGBP=X",
    "NZD/USD": "NZDUSD=X", "USD/CHF": "USDCHF=X"
}

TIMEFRAMES = {
    "M15 (15 min)": {"interval": "15m", "period": "1mo"},
    "M5 (5 min)": {"interval": "5m", "period": "5d"},
    "M30 (30 min)": {"interval": "30m", "period": "1mo"},
    "H1 (1 heure)": {"interval": "1h", "period": "3mo"}
}

@st.cache_data(ttl=45, show_spinner=False)
def fetch_data(symbol, period, interval):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval, timeout=5)
        if df is None or df.empty or len(df) < 35:
            return None
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        return df
    except Exception:
        return None
        # ============================================================
# MOTEUR D'ANALYSE v5.2 (MATRICE DE CONFLUENCE)
# ============================================================
def analyze_market(symbol_name, df_ltf, df_htf, mode):
    if df_ltf is None or len(df_ltf) < 35:
        return None

    df = df_ltf.copy()

    # Indicateurs LTF
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    diff = df['Close'].diff()
    gain = diff.clip(lower=0).rolling(14).mean()
    loss = (-diff.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift()))
    )
    df['ATR'] = tr.rolling(14).mean()

    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr14 = df['ATR']
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-9))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(14).mean()

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']

    # Clôture N-2 (Zéro repeinte)
    c = df.iloc[-2]
    prev = df.iloc[-3]
    atr_val = c['ATR'] if not np.isnan(c['ATR']) else (c['High'] - c['Low'])
    adx_val = c['ADX'] if not np.isnan(c['ADX']) else 25

    total_range = c['High'] - c['Low']
    body = abs(c['Close'] - c['Open'])
    lower_wick = min(c['Close'], c['Open']) - c['Low']
    upper_wick = c['High'] - max(c['Close'], c['Open'])

    signal_time = str(df.index[-2])
    now_h = datetime.now(timezone.utc).hour
    is_session_active = (7 <= now_h < 20)

    # 1. Analyse HTF (H1)
    htf_trend = "Neutre"
    if df_htf is not None and len(df_htf) > 50:
        htf_e20 = df_htf['Close'].ewm(span=20, adjust=False).mean().iloc[-2]
        htf_e50 = df_htf['Close'].ewm(span=50, adjust=False).mean().iloc[-2]
        if htf_e20 > htf_e50:
            htf_trend = "Haussiere"
        elif htf_e20 < htf_e50:
            htf_trend = "Baissiere"

    # Pente EMA50
    if len(df) >= 7:
        ema50_slope = df['EMA50'].iloc[-2] - df['EMA50'].iloc[-7]
    else:
        ema50_slope = 0
    is_trending_up = ema50_slope > (0.05 * atr_val)
    is_trending_down = ema50_slope < -(0.05 * atr_val)

    atr_pct = (atr_val / c['Close']) * 100 if c['Close'] > 0 else 0
    atr_ok = atr_pct > 0.02
    is_ranging = adx_val < 22

    # Price Action
    if total_range > 0:
        is_bull_pinbar = (lower_wick >= 0.6 * total_range) and (body <= 0.3 * total_range)
        is_bear_pinbar = (upper_wick >= 0.6 * total_range) and (body <= 0.3 * total_range)
    else:
        is_bull_pinbar = is_bear_pinbar = False

    is_bull_engulf = (c['Close'] > prev['High']) and (c['Close'] > c['Open'])
    is_bear_engulf = (c['Close'] < prev['Low']) and (c['Close'] < c['Open'])

    trend_up = c['EMA20'] > c['EMA50'] and c['Close'] > c['EMA50']
    trend_down = c['EMA20'] < c['EMA50'] and c['Close'] < c['EMA50']

    in_buy_zone = c['Low'] <= c['EMA20'] * 1.001
    in_sell_zone = c['High'] >= c['EMA20'] * 0.999

    macd_bull = c['Hist'] > prev['Hist']
    macd_bear = c['Hist'] < prev['Hist']

    # ============================================================
    # EVALUATION DES 5 PILIERS DE CONFLUENCE
    # ============================================================
    piliers_buy = {
        "1. Tendance H1": htf_trend == "Haussiere",
        "2. Structure LTF": trend_up and is_trending_up,
        "3. Zone Retest EMA": in_buy_zone,
        "4. Price Action": is_bull_pinbar or is_bull_engulf,
        "5. Momentum (RSI/MACD)": (40 <= c['RSI'] <= 65) and macd_bull
    }

    piliers_sell = {
        "1. Tendance H1": htf_trend == "Baissiere",
        "2. Structure LTF": trend_down and is_trending_down,
        "3. Zone Retest EMA": in_sell_zone,
        "4. Price Action": is_bear_pinbar or is_bear_engulf,
        "5. Momentum (RSI/MACD)": (35 <= c['RSI'] <= 60) and macd_bear
    }

    conf_buy_score = sum(piliers_buy.values())
    conf_sell_score = sum(piliers_sell.values())

    # Calcul du score pondéré %
    score_buy = (conf_buy_score / 5.0) * 100
    score_sell = (conf_sell_score / 5.0) * 100

    # Gestion des pips (JPY vs Standards)
    is_jpy = "JPY" in symbol_name
    pip_multiplier = 100.0 if is_jpy else 10000.0
    min_sl_dist = 0.180 if is_jpy else 0.00180  # 18 pips plancher

    entry = sl = tp = None
    risk = 0

    if trend_up:
        entry = df.iloc[-1]['Open']
        recent_low = df['Low'].iloc[-7:-2].min()
        sl = round(min(recent_low, entry - (1.5 * atr_val)), 5 if not is_jpy else 3)
        risk = entry - sl
        if risk < min_sl_dist:
            sl = round(entry - min_sl_dist, 5 if not is_jpy else 3)
            risk = min_sl_dist
        tp = round(entry + (2.0 * risk), 5 if not is_jpy else 3)

    elif trend_down:
        entry = df.iloc[-1]['Open']
        recent_high = df['High'].iloc[-7:-2].max()
        sl = round(max(recent_high, entry + (1.5 * atr_val)), 5 if not is_jpy else 3)
        risk = sl - entry
        if risk < min_sl_dist:
            sl = round(entry + min_sl_dist, 5 if not is_jpy else 3)
            risk = min_sl_dist
        tp = round(entry - (2.0 * risk), 5 if not is_jpy else 3)

    # Seuils de validation selon la sensibilité
    if "Strict" in mode:
        min_confluence = 4  # Uniquement Setup A ou A+ (4/5 ou 5/5)
    elif "Modere" in mode:
        min_confluence = 3  # Setups 3/5 acceptés
    else:
        min_confluence = 2

    verdict = "ATTENDRE"
    color = "#FFA500"
    action = "HOLD"
    final_conf = max(conf_buy_score, conf_sell_score)
    final_score = int(max(score_buy, score_sell))
    piliers_actifs = piliers_buy if conf_buy_score >= conf_sell_score else piliers_sell

    # Détermination du grade
    if final_conf == 5:
        grade = "⭐⭐⭐⭐⭐ A+ (5/5)"
    elif final_conf == 4:
        grade = "⭐⭐⭐⭐ A (4/5)"
    elif final_conf == 3:
        grade = "⭐⭐⭐ B (3/5)"
    else:
        grade = "⚪ C (" + str(final_conf) + "/5)"

    # Raisonnement décisionnel
    if not is_session_active:
        motif_final = "KILL-ZONE : Session Asie/Nuit. Trades bloqués."
    elif not atr_ok:
        motif_final = "Volatilité trop faible (ATR). Marché endormi."
    elif is_ranging:
        motif_final = "Marché en range (ADX < 22). Pas de direction."
    elif not is_trending_up and not is_trending_down:
        motif_final = "EMA50 plate : marché plat. Attendre impulsion."
    elif conf_buy_score >= min_confluence and conf_buy_score > conf_sell_score and htf_trend != "Baissiere":
        verdict = "ACHAT (" + str(final_score) + "%)"
        color = "#00FF88"
        action = "BUY"
        motif_final = "Setup " + grade + " validé avec " + str(conf_buy_score) + "/5 confluences."
    elif conf_sell_score >= min_confluence and conf_sell_score > conf_buy_score and htf_trend != "Haussiere":
        verdict = "VENTE (" + str(final_score) + "%)"
        color = "#FF3366"
        action = "SELL"
        motif_final = "Setup " + grade + " validé avec " + str(conf_sell_score) + "/5 confluences."
    else:
        motif_final = "Confluence insuffisante (" + str(final_conf) + "/5). " + str(min_confluence) + "/5 requis en mode " + mode.split()[0] + "."

    return {
        "verdict": verdict, "color": color, "action": action, "motif": motif_final,
        "entry": entry, "sl": sl, "tp": tp, "score": final_score,
        "confluence_score": final_conf, "confluence_grade": grade, "piliers": piliers_actifs,
        "rsi": round(c['RSI'], 1), "atr": round(atr_val, 5), "adx": round(adx_val, 1),
        "htf_trend": htf_trend, "trend": "Haussiere" if trend_up else ("Baissiere" if trend_down else "Range"),
        "price": df.iloc[-1]['Close'], "signal_time": signal_time,
        "risk_pips": round(risk * pip_multiplier, 1) if entry and sl else 0
    }
    # ============================================================
# PAGE 1 : TABLEAU DE BORD (SCANNER DE CONFLUENCE)
# ============================================================
if page == "📊 Tableau de bord":
    st.markdown("# 📊 Tableau de Bord Multi-Paires")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        try:
            test = yf.Ticker("AUDUSD=X")
            h = test.history(period="1d", interval="1h", timeout=4)
            if not h.empty:
                st.success("🟢 Yahoo Finance : Connecté")
            else:
                st.error("🔴 Yahoo Finance : Pas de données")
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
    st.markdown("### Scanner des Paires & Confluence")
    tf_scan = st.selectbox("Unité de temps LTF", list(TIMEFRAMES.keys()), index=0)
    tf_cfg = TIMEFRAMES[tf_scan]

    if st.button("🚀 Scanner le Marché", use_container_width=True, type="primary"):
        results = []
        bar = st.progress(0)
        for i, (name, sym) in enumerate(PAIRS.items()):
            df_ltf = fetch_data(sym, tf_cfg["period"], tf_cfg["interval"])
            df_htf = fetch_data(sym, "3mo", "1h")
            res = analyze_market(name, df_ltf, df_htf, sensibilite)
            if res:
                if res["action"] in ["BUY", "SELL"]:
                    ajouter_signal(
                        name, res["verdict"], res["action"],
                        res["entry"], res["sl"], res["tp"],
                        tf_scan, res["motif"], res["score"],
                        res["confluence_grade"], res["signal_time"]
                    )
                results.append({
                    "Paire": name, "Verdict": res["verdict"], "Confluence": res["confluence_grade"],
                    "H1": res["htf_trend"], "RSI": res["rsi"], "ADX": res["adx"],
                    "SL (pips)": res["risk_pips"], "Signal à": res["signal_time"], "Analyse": res["motif"]
                })
            bar.progress((i + 1) / len(PAIRS))

        if results:
            for r in results:
                if "ACHAT" in r["Verdict"]:
                    st.success("🟢 **" + r["Paire"] + "** | " + r["Verdict"] + " | " + r["Confluence"] + " | SL: " + str(r["SL (pips)"]) + " pips | " + r["Analyse"])
                elif "VENTE" in r["Verdict"]:
                    st.error("🔴 **" + r["Paire"] + "** | " + r["Verdict"] + " | " + r["Confluence"] + " | SL: " + str(r["SL (pips)"]) + " pips | " + r["Analyse"])
                else:
                    st.warning("🟡 **" + r["Paire"] + "** | " + r["Verdict"] + " | " + r["Confluence"] + " | " + r["Analyse"])
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

# ============================================================
# PAGE 2 : ANALYSE & CHECKLIST DES 5 PILIERS
# ============================================================
elif page == "🔍 Analyse Confluence":
    st.markdown("# 🔍 Analyse Détaillée & Confluence")
    st.markdown("---")

    col1, col2 = st.columns(2)
    p_name = col1.selectbox("Paire", list(PAIRS.keys()), index=0)
    tf_name = col2.selectbox("Unité de temps", list(TIMEFRAMES.keys()), index=0)

    tf_cfg = TIMEFRAMES[tf_name]
    df_ltf = fetch_data(PAIRS[p_name], tf_cfg["period"], tf_cfg["interval"])
    df_htf = fetch_data(PAIRS[p_name], "3mo", "1h")
    res = analyze_market(p_name, df_ltf, df_htf, sensibilite)

    if res:
        st.markdown(
            "<div style='background-color: #1E222D; border-left: 8px solid " + res["color"] + "; padding: 20px; border-radius: 8px; margin-bottom: 15px;'>"
            + "<h1 style='color: " + res["color"] + "; margin: 0;'>" + res["verdict"] + "</h1>"
            + "<h3 style='color: #EEE; margin: 5px 0 0 0;'>Grade : " + res["confluence_grade"] + "</h3>"
            + "<p style='color: #CCC; margin: 8px 0 0 0;'><b>" + p_name + "</b> | " + tf_name + " | Prix : <b>" + str(round(res["price"], 5)) + "</b> | Tendance H1 : <b>" + res["htf_trend"] + "</b></p>"
            + "<p style='color: #AAA; margin: 5px 0 0 0;'><b>Observation :</b> " + res["motif"] + "</p>"
            + "<p style='color: #666; font-size: 12px; margin: 5px 0 0 0;'>Bougie clôturée à : " + res["signal_time"] + "</p></div>",
            unsafe_allow_html=True
        )

        st.markdown("### 🧩 Matrice des 5 Piliers de Confluence")
        p_cols = st.columns(5)
        for idx, (nom_pilier, etat) in enumerate(res["piliers"].items()):
            with p_cols[idx]:
                if etat:
                    st.success("🟢 " + nom_pilier + "\n\n**Validé**")
                else:
                    st.error("🔴 " + nom_pilier + "\n\n**Non aligné**")

        st.markdown("---")
        if res["action"] in ["BUY", "SELL"]:
            st.markdown("### 📋 Ordre d'Exécution MT5")
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Prix Entrée", str(round(res["entry"], 5)))
            x2.metric("Stop Loss (" + str(res["risk_pips"]) + " pips)", str(round(res["sl"], 5)))
            x3.metric("Take Profit", str(round(res["tp"], 5)))
            x4.metric("Ratio R:R", "1 : 2.0")

            if st.button("📝 Sauvegarder dans le Journal", use_container_width=True, type="primary"):
                ok = ajouter_signal(
                    p_name, res["verdict"], res["action"],
                    res["entry"], res["sl"], res["tp"],
                    tf_name, res["motif"], res["score"],
                    res["confluence_grade"], res["signal_time"]
                )
                if ok:
                    st.success("Signal enregistré dans le journal permanent avec verrouillage 3 bougies !")
                else:
                    st.warning("Signal déjà présent ou verrouillage actif pour cette unité de temps.")
        else:
            st.info("💡 Attendez que 4 ou 5 confluences soient alignées avant d'entrer sur le marché.")

# ============================================================
# PAGES SECONDAIRES
# ============================================================
elif page == "📈 Graphique":
    st.markdown("# 📈 Graphique des Prix")
    p_name = st.selectbox("Paire", list(PAIRS.keys()))
    tf_name = st.selectbox("Unité de temps", list(TIMEFRAMES.keys()))
    df = fetch_data(PAIRS[p_name], TIMEFRAMES[tf_name]["period"], TIMEFRAMES[tf_name]["interval"])
    if df is not None:
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Prix"))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#00E5FF', width=1.5), name="EMA 20"))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#FF9100', width=1.5), name="EMA 50"))
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=550)
        st.plotly_chart(fig, use_container_width=True)

elif page == "⚠️ Risque":
    st.markdown("# ⚠️ Calculateur de Position MT5")
    cap = st.number_input("Capital du Compte ($)", value=1000.0, step=100.0)
    risk_pct = st.number_input("Risque par Trade (%)", value=1.0, step=0.5)
    sl_pips = st.number_input("Stop Loss (pips)", value=20.0, step=1.0)
    
    montant_risque = cap * (risk_pct / 100.0)
    lot_calcule = round(montant_risque / (sl_pips * 10.0), 2)
    
    c1, c2 = st.columns(2)
    c1.metric("Montant Risqué", "$" + str(round(montant_risque, 2)))
    c2.metric("Taille de Lot MT5", str(lot_calcule) + " Lot(s)")

elif page == "📓 Journal":
    st.markdown("# 📓 Journal de Trading Persistant")
    st.session_state.journal = charger_journal()
    if len(st.session_state.journal) == 0:
        st.info("Aucun signal enregistré.")
    else:
        df_j = pd.DataFrame(st.session_state.journal)
        st.dataframe(df_j, use_container_width=True, hide_index=True)
        c_dl, c_clr = st.columns(2)
        with c_dl:
            st.download_button("📥 Télécharger CSV", df_j.to_csv(index=False).encode('utf-8'), "forexscope_journal.csv", "text/csv", use_container_width=True)
        with c_clr:
            if st.button("🗑️ Effacer le journal", use_container_width=True):
                st.session_state.journal = []
                sauvegarder_journal([])
                st.rerun()

elif page == "⚙️ Parametres":
    st.markdown("# ⚙️ Paramètres & Fonctionnement")
    st.markdown("""
    ### 🛡️ Règles de Sécurité Actives :
    - **Score de Confluence (5 Piliers) :** Exige l'accord entre Tendance H1, Structure M15, Retest EMA, Price Action et Momentum.
    - **Grade A+ (5/5) & A (4/5) :** Priorise uniquement les configurations à très fort avantage statistique.
    - **Stop Loss Structurel avec Plancher 18 Pips :** Protège contre les mèches de spread.
    - **Verrouillage Dynamique 3 Bougies :** Empêche les faux signaux successifs sur la même paire.
    - **Calcul Strict sur Bougie Clôturée N-2 :** Zéro repeinte.
    """)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timezone

st.set_page_config(page_title="ForexScope Pro", page_icon="🎯", layout="wide")

# ============================================================
# INITIALISATION DU JOURNAL (Session State)
# ============================================================
if "journal" not in st.session_state:
    st.session_state.journal = []

def ajouter_au_journal(paire, verdict, action, entry, sl, tp, timeframe, motif):
    if action in ["BUY", "SELL"]:
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
nb_signaux = len([s for s in st.session_state.journal if s["Statut"] == "Actif"])
st.sidebar.caption(f"📓 Journal : {len(st.session_state.journal)} signaux")
st.sidebar.caption(f"🟢 Actifs : {nb_signaux}")
st.sidebar.caption("Source : Yahoo Finance")
st.sidebar.caption("v3.0 - Pullback/Rejet")

# ============================================================
# CONFIGURATION
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
# FONCTION DE CONNEXION YAHOO FINANCE
# ============================================================
def tester_connexion_yahoo():
    try:
        test = yf.Ticker("AUDUSD=X")
        hist = test.history(period="1d", interval="1h")
        if hist.empty:
            return False, "Aucune donnee recue"
        return True, "Connecte"
    except Exception as e:
        return False, str(e)

# ============================================================
# FONCTION DE DONNEES
# ============================================================
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
# MOTEUR D'ANALYSE
# ============================================================
def analyze_market(df):
    if df is None or len(df) < 60:
        return None

    df = df.copy()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']

    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = df['ATR']
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-9))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(14).mean()

    c = df.iloc[-2]
    prev = df.iloc[-3]
    atr_val = c['ATR']

    body = abs(c['Close'] - c['Open'])
    upper_wick = c['High'] - max(c['Close'], c['Open'])
    lower_wick = min(c['Close'], c['Open']) - c['Low']

    trend_up = c['EMA20'] > c['EMA50'] and c['Close'] > c['EMA50']
    trend_down = c['EMA20'] < c['EMA50'] and c['Close'] < c['EMA50']
    is_ranging = c['ADX'] < 20 if not np.isnan(c['ADX']) else False

    in_buy_zone = c['Low'] <= c['EMA20'] * 1.001 or c['Low'] <= c['EMA50'] * 1.001
    in_sell_zone = c['High'] >= c['EMA20'] * 0.999 or c['High'] >= c['EMA50'] * 0.999

    bullish_reject = lower_wick >= (1.3 * body) or (c['Close'] > prev['High'] and c['Close'] > c['Open'])
    bearish_reject = upper_wick >= (1.3 * body) or (c['Close'] < prev['Low'] and c['Close'] < c['Open'])

    macd_bull = c['MACD'] > c['Signal'] or c['Histogram'] > df.iloc[-3]['Histogram']
    macd_bear = c['MACD'] < c['Signal'] or c['Histogram'] < df.iloc[-3]['Histogram']

    verdict = "ATTENDRE"
    color = "#FFA500"
    action = "HOLD"
    motif = "Pas de configuration valide."
    entry = sl = tp = None
    score = 0

    if is_ranging:
        motif = "Marche en range (ADX < 20). Pas de tendance claire."
    elif trend_up and in_buy_zone and bullish_reject and (35 <= c['RSI'] <= 65):
        score = 70
        if macd_bull:
            score = 85
        verdict = "ACHAT (BUY)"
        color = "#00FF88"
        action = "BUY"
        motif = "Tendance haussiere + Pullback EMA + Rejet + RSI sain."
        entry = df.iloc[-1]['Open']
        sl = round(c['Low'] - (0.5 * atr_val), 5)
        risk = entry - sl
        tp = round(entry + (2 * risk), 5) if risk > 0 else round(entry + (2 * atr_val), 5)
    elif trend_down and in_sell_zone and bearish_reject and (35 <= c['RSI'] <= 65):
        score = -70
        if macd_bear:
            score = -85
        verdict = "VENTE (SELL)"
        color = "#FF3366"
        action = "SELL"
        motif = "Tendance baissiere + Retest EMA + Rejet + RSI sain."
        entry = df.iloc[-1]['Open']
        sl = round(c['High'] + (0.5 * atr_val), 5)
        risk = sl - entry
        tp = round(entry - (2 * risk), 5) if risk > 0 else round(entry - (2 * atr_val), 5)
    elif trend_up and not in_buy_zone:
        motif = "Tendance haussiere mais prix trop etendu. Attendre pullback."
        score = 30
    elif trend_down and not in_sell_zone:
        motif = "Tendance baissiere mais prix trop etendu. Attendre retest."
        score = -30

    return {
        "verdict": verdict, "color": color, "action": action,
        "motif": motif, "entry": entry, "sl": sl, "tp": tp,
        "score": score, "rsi": round(c['RSI'], 1),
        "atr": round(atr_val, 5),
        "adx": round(c['ADX'], 1) if not np.isnan(c['ADX']) else 0,
        "macd": round(c['MACD'], 5),
        "ema20": round(c['EMA20'], 5),
        "ema50": round(c['EMA50'], 5),
        "trend": "Haussiere" if trend_up else ("Baissiere" if trend_down else "Range"),
        "closed_time": str(df.index[-2])
    }

# ============================================================
# PAGE 1 : TABLEAU DE BORD
# ============================================================
if page == "📊 Tableau de bord":
    st.markdown("# 📊 Tableau de Bord")

    # --- STATUT DE CONNEXION ---
    st.markdown("---")
    col_statut, col_heure, col_refresh = st.columns([2, 2, 1])

    with col_statut:
        with st.spinner("Test de connexion..."):
            connected, msg = tester_connexion_yahoo()
        if connected:
            st.success("🟢 Yahoo Finance : Connecte")
        else:
            st.error(f"🔴 Yahoo Finance : Deconnecte ({msg})")

    with col_heure:
        heure_actuelle = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        st.info(f"🕐 Derniere analyse : {heure_actuelle}")

    with col_refresh:
        if st.button("🔄 Rafraichir", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # --- SCANNER DE PAIRES ---
    st.markdown("### Scanner multi-paires")
    tf_scan = st.selectbox("Unite de temps pour le scan", list(TIMEFRAMES.keys()), index=1)
    tf_cfg = TIMEFRAMES[tf_scan]

    if st.button("🚀 Lancer le scan complet", use_container_width=True, type="primary"):
        results = []
        progress = st.progress(0)
        for i, (name, yahoo_sym) in enumerate(PAIRS.items()):
            df = fetch_data(yahoo_sym, tf_cfg["period"], tf_cfg["interval"])
            res = analyze_market(df)
            if res:
                if res["action"] in ["BUY", "SELL"]:
                    ajouter_au_journal(
                        name, res["verdict"], res["action"],
                        res["entry"], res["sl"], res["tp"], tf_scan, res["motif"]
                    )
                results.append({
                    "Paire": name,
                    "Verdict": res["verdict"],
                    "Tendance": res["trend"],
                    "RSI": res["rsi"],
                    "ADX": res["adx"],
                    "Score": res["score"],
                    "ATR": res["atr"]
                })
            progress.progress((i + 1) / len(PAIRS))

        if results:
            st.markdown("---")
            st.markdown("### Resultats")
            for _, row in pd.DataFrame(results).iterrows():
                if "ACHAT" in row["Verdict"]:
                    icon = "🟢"
                elif "VENTE" in row["Verdict"]:
                    icon = "🔴"
                else:
                    icon = "🟡"
                st.markdown(f"{icon} **{row['Paire']}** | {row['Verdict']} | Tendance: {row['Tendance']} | RSI: {row['RSI']} | ADX: {row['ADX']}")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        else:
            st.warning("Aucune donnee disponible.")

# ============================================================
# PAGE 2 : ANALYSE DETAILLEE
# ============================================================
elif page == "🔍 Analyse":
    st.markdown("# 🔍 Analyse Detaillee")
    st.markdown("Moteur Pullback/Rejet non-repainting sur bougie cloturee.")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        pair_name = st.selectbox("Paire", list(PAIRS.keys()), index=0)
    with col2:
        tf_name = st.selectbox("Unite de temps", list(TIMEFRAMES.keys()), index=1)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    tf_cfg = TIMEFRAMES[tf_name]
    yahoo_sym = PAIRS[pair_name]

    with st.spinner("Analyse en cours..."):
        df = fetch_data(yahoo_sym, tf_cfg["period"], tf_cfg["interval"])
        res = analyze_market(df)

    if res:
        current_price = df.iloc[-1]['Close']

        st.markdown(f"""
        <div style='background-color: #1E222D; border-left: 8px solid {res["color"]}; padding: 20px; border-radius: 8px;'>
            <h1 style='color: {res["color"]}; margin: 0;'>{res["verdict"]}</h1>
            <p style='color: #CCC; margin: 8px 0 0 0;'><b>{pair_name}</b> | {tf_name} | Prix: {current_price:.5f}</p>
            <p style='color: #AAA; margin: 5px 0 0 0;'>{res["motif"]}</p>
            <p style='color: #666; font-size: 12px; margin: 5px 0 0 0;'>Bougie analysee : {res["closed_time"]}</p>
        </div>
        """, unsafe_allow_html=True)

        if res["action"] in ["BUY", "SELL"]:
            st.markdown("### 📋 Parametres d'execution")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Entree", f"{res['entry']:.5f}")
            c2.metric("Stop Loss", f"{res['sl']:.5f}")
            c3.metric("Take Profit", f"{res['tp']:.5f}")
            c4.metric("Ratio R:R", "1 : 2.0")

            if st.button("📝 Enregistrer ce signal dans le Journal", use_container_width=True, type="primary"):
                ajouter_au_journal(
                    pair_name, res["verdict"], res["action"],
                    res["entry"], res["sl"], res["tp"], tf_name, res["motif"]
                )
                st.success("Signal enregistre dans le journal !")

            st.success("Ouvre MT5 et place ton ordre avec ces niveaux exacts.")
        else:
            st.info("Pas de trade pour le moment. Preserver son capital est une strategie gagnante.")

        st.markdown("### 📊 Indicateurs")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tendance", res["trend"])
        c2.metric("RSI (14)", res["rsi"])
        c3.metric("ADX", res["adx"])
        c4.metric("MACD", res["macd"])
        c5.metric("ATR", res["atr"])
    else:
        st.error("Donnees insuffisantes pour cette paire/unite de temps.")

# ============================================================
# PAGE 3 : GRAPHIQUE
# ============================================================
elif page == "📈 Graphique":
    st.markdown("# 📈 Graphique Interactif")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        pair_name = st.selectbox("Paire", list(PAIRS.keys()), index=0, key="graph_pair")
    with col2:
        tf_name = st.selectbox("Unite de temps", list(TIMEFRAMES.keys()), index=1, key="graph_tf")

    tf_cfg = TIMEFRAMES[tf_name]
    yahoo_sym = PAIRS[pair_name]
    df = fetch_data(yahoo_sym, tf_cfg["period"], tf_cfg["interval"])

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
        fig.update_layout(
            template="plotly_dark", title=f"{pair_name} - {tf_name}",
            xaxis_rangeslider_visible=False, height=550,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='#BB86FC', width=1.5), name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
        fig_rsi.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.05)
        fig_rsi.update_layout(
            template="plotly_dark", title="RSI (14)", height=200,
            margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
    else:
        st.warning("Donnees indisponibles pour ce graphique.")

# ============================================================
# PAGE 4 : GESTION DU RISQUE
# ============================================================
elif page == "⚠️ Risque":
    st.markdown("# ⚠️ Calculateur de Risque")
    st.markdown("Calcule la taille de lot ideale pour ne jamais perdre plus que ton pourcentage defini.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        capital = st.number_input("Capital du compte (USD)", min_value=10.0, value=1000.0, step=100.0)
    with c2:
        risk_pct = st.number_input("Risque par trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    with c3:
        sl_pips = st.number_input("Stop Loss (en pips)", min_value=1.0, value=20.0, step=1.0)

    st.markdown("---")
    risk_amount = capital * (risk_pct / 100)
    pip_value_standard = 10.0
    lot_size = risk_amount / (sl_pips * pip_value_standard)

    st.markdown("### Resultat")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Montant a risquer", f"${risk_amount:.2f}")
    rc2.metric("Taille de lot recommandee", f"{lot_size:.2f} lots")
    rc3.metric("Perte max si SL touche", f"-${risk_amount:.2f}")

    st.markdown("---")
    st.markdown("### 📏 Tableau de reference rapide")
    table_data = []
    for sl in [10, 15, 20, 25, 30, 50]:
        lot = risk_amount / (sl * pip_value_standard)
        table_data.append({"Stop Loss (pips)": sl, "Taille de lot": round(lot, 2), "Risque ($)": round(risk_amount, 2)})
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    st.info("Regle d'or : Ne risque jamais plus de 1 a 2% de ton capital sur un seul trade.")

# ============================================================
# PAGE 5 : JOURNAL DE TRADING
# ============================================================
elif page == "📓 Journal":
    st.markdown("# 📓 Journal de Trading")
    st.markdown("Historique complet de tous les signaux emis par ForexScope.")
    st.markdown("---")

    if len(st.session_state.journal) == 0:
        st.info("📭 Le journal est vide. Lance une analyse ou un scan pour generer des signaux.")
        st.caption("Les signaux ACHAT et VENTE seront automatiquement enregistres ici avec leur horodatage, SL et TP.")
    else:
        # Statistiques rapides
        total = len(st.session_state.journal)
        buys = len([s for s in st.session_state.journal if "BUY" in s["Verdict"]])
        sells = len([s for s in st.session_state.journal if "SELL" in s["Verdict"]])

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Total Signaux", total)
        sc2.metric("Signaux ACHAT", buys)
        sc3.metric("Signaux VENTE", sells)

        st.markdown("---")

        # Tableau du journal
        df_journal = pd.DataFrame(st.session_state.journal)
        st.dataframe(df_journal, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Actions
        col_exp, col_clear = st.columns(2)
        with col_exp:
            csv = df_journal.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exporter le journal en CSV",
                data=csv,
                file_name=f"forexscope_journal_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_clear:
            if st.button("🗑️ Vider le journal", use_container_width=True):
                st.session_state.journal = []
                st.rerun()

        st.caption("Note : Le journal est reinitialise a chaque fermeture de l'application. Exporte regulierement en CSV pour conserver ton historique.")

# ============================================================
# PAGE 6 : PARAMETRES
# =================================


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# 1. CONFIGURATION DE L'APPLICATION
# =========================================================

st.set_page_config(
    page_title="ForexScope",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. STYLE DE L'INTERFACE
# =========================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 16px;
        border-radius: 12px;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. MARCHÉS DISPONIBLES
# =========================================================

SYMBOLS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "Or (Gold)": "GC=F",
    "Argent (Silver)": "SI=F",
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Bitcoin / USD": "BTC-USD",
}

# Yahoo Finance ne fournit pas de bougies M1 pour une
# période longue. Les données intraday ont des limites.
INTERVALS = {
    "M1": {"interval": "1m", "period": "7d"},
    "M5": {"interval": "5m", "period": "60d"},
    "M15": {"interval": "15m", "period": "60d"},
    "M30": {"interval": "30m", "period": "60d"},
    "H1": {"interval": "60m", "period": "60d"},
    "H4": {"interval": "60m", "period": "60d"},
    "D1": {"interval": "1d", "period": "2y"},
}

# =========================================================
# 4. VALIDATION DES BOUGIES
# =========================================================

def validate_ohlc(data):
    """Nettoie et valide les données OHLC reçues."""

    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()

    # Compatibilité avec certaines versions de yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    required = ["open", "high", "low", "close"]

    if not all(column in data.columns for column in required):
        return pd.DataFrame()

    if "time" not in data.columns:
        data["time"] = data.index

    for column in required:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data["time"] = pd.to_datetime(
        data["time"], errors="coerce", utc=True
    )

    data = data.dropna(
        subset=["time", "open", "high", "low", "close"]
    )

    data = data[
        (data["high"] >= data["low"])
        & (data["high"] >= data["open"])
        & (data["high"] >= data["close"])
        & (data["low"] <= data["open"])
        & (data["low"] <= data["close"])
        & (data["close"] > 0)
    ]

    data = (
        data.sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )

    return data[["time", "open", "high", "low", "close"]]


# =========================================================
# 5. RÉCUPÉRATION DES DONNÉES YAHOO FINANCE
# =========================================================

@st.cache_data(ttl=30, show_spinner=False)
def fetch_candles(symbol, timeframe, count=300):
    """Télécharge les bougies puis renvoie des données OHLC valides."""

    if timeframe not in INTERVALS:
        return pd.DataFrame()

    settings = INTERVALS[timeframe]

    try:
        data = yf.download(
            tickers=symbol,
            period=settings["period"],
            interval=settings["interval"],
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )

        if data is None or data.empty:
            return pd.DataFrame()

        # Aplatit les colonnes si Yahoo renvoie un MultiIndex.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = validate_ohlc(data)

        if data.empty:
            return pd.DataFrame()

        # Construction des bougies H4 à partir des bougies H1.
        if timeframe == "H4":
            data = data.set_index("time")

            data = data.resample(
                "4h",
                origin="start_day",
                label="left",
                closed="left",
            ).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            })

            data = data.dropna().reset_index()
            data = validate_ohlc(data)

        return data.tail(int(count)).reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


# =========================================================
# 6. FONCTIONS D'AFFICHAGE
# =========================================================

def format_price(price):
    """Adapte l'affichage du prix à son ordre de grandeur."""

    if pd.isna(price):
        return "—"

    if price >= 1000:
        return f"{price:,.2f}"

    if price >= 10:
        return f"{price:.3f}"

    return f"{price:.5f}"


def build_candlestick_chart(data, market_name, timeframe):
    """Construit le graphique en chandeliers."""

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data["time"],
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        name=market_name,
        increasing_line_color="#16a34a",
        decreasing_line_color="#dc2626",
    ))

    fig.update_layout(
        title=f"{market_name} — {timeframe}",
        xaxis_title="Date et heure (UTC)",
        yaxis_title="Prix",
        template="plotly_dark",
        height=550,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h"),
    )

    return fig


# =========================================================
# 7. BARRE LATÉRALE
# =========================================================

with st.sidebar:
    st.title("ForexScope")
    st.caption("Analyse des marchés financiers")

    st.divider()

    market_name = st.selectbox(
        "Marché",
        options=list(SYMBOLS.keys()),
        index=0,
    )

    timeframe = st.selectbox(
        "Unité de temps",
        options=list(INTERVALS.keys()),
        index=2,
    )

    candle_count = st.slider(
        "Nombre de bougies",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
    )

    refresh = st.button(
        "Actualiser les données",
        use_container_width=True,
    )

    if refresh:
        st.cache_data.clear()

    st.divider()

    st.caption(
        "Source : Yahoo Finance. "
        "Disponibilité et délais variables selon le marché."
    )


# =========================================================
# 8. TABLEAU DE BORD PRINCIPAL
# =========================================================

st.title("ForexScope")
st.caption(
    "Plateforme d'analyse technique et de suivi des marchés."
)

tab_market, tab_analysis, tab_history, tab_settings = st.tabs([
    "📈 Marché",
    "🎯 Analyse",
    "📋 Historique",
    "⚙️ Paramètres",
])

with st.spinner(f"Chargement des données : {market_name}..."):
    candles = fetch_candles(
        SYMBOLS[market_name],
        timeframe,
        candle_count,
    )

with tab_market:
    if candles.empty:
        st.error(
            "Aucune bougie reçue. Vérifie la connexion Internet, "
            "le symbole et l'intervalle choisi."
        )

        st.info(
            "Yahoo Finance limite certaines données intraday. "
            "Cela ne signifie pas nécessairement que le marché est fermé."
        )

    else:
        latest = candles.iloc[-1]

        previous_close = (
            candles.iloc[-2]["close"]
            if len(candles) >= 2
            else latest["close"]
        )

        change = latest["close"] - previous_close

        change_pct = (
            change / previous_close * 100
            if previous_close != 0
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Marché",
            market_name,
        )

        col2.metric(
            "Dernier prix",
            format_price(latest["close"]),
        )

        col3.metric(
            "Variation de la dernière bougie",
            format_price(change),
            f"{change_pct:.3f}%",
        )

        col4.metric(
            "Bougies disponibles",
            f"{len(candles)}",
        )

        st.plotly_chart(
            build_candlestick_chart(
                candles, market_name, timeframe
            ),
            use_container_width=True,
        )

        with st.expander("Consulter les données OHLC"):
            st.dataframe(
                candles.sort_values(
                    "time", ascending=False
                ),
                use_container_width=True,
                hide_index=True,
            )

with tab_analysis:
    st.subheader("Analyse technique")

    st.info(
        "Le moteur d'analyse sera ajouté dans la partie 2. "
        "Aucun signal d'achat ou de vente n'est encore généré."
    )

    if not candles.empty:
        st.write(f"Marché sélectionné : **{market_name}**")
        st.write(f"Unité de temps : **{timeframe}**")
        st.write(
            f"Dernière bougie (UTC) : "
            f"**{candles.iloc[-1]['time']}**"
        )

with tab_history:
    st.subheader("Historique")

    st.info(
        "L'enregistrement des signaux et l'historique "
        "seront ajoutés après la validation du moteur d'analyse."
    )

with tab_settings:
    st.subheader("Paramètres de données")

    st.write(f"Fournisseur : Yahoo Finance")
    st.write(f"Symbole Yahoo : {SYMBOLS[market_name]}")
    st.write(f"Unité sélectionnée : {timeframe}")
    st.write(f"Nombre de bougies demandé : {candle_count}")

    st.warning(
        "Les données peuvent être retardées, incomplètes "
        "ou indisponibles. Les bougies H4 sont reconstruites "
        "à partir des bougies H1."
    )

st.divider()

st.caption(
    f"ForexScope • Mis à jour à l'ouverture de cette page : "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)


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
# 5B. INDICATEURS TECHNIQUES
# =========================================================

def calculate_indicators(data):
    """Calcule les indicateurs utilisés par ForexScope."""

    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Moyennes mobiles exponentielles
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["ema_200"] = close.ewm(span=200, adjust=False).mean()

    # RSI sur 14 périodes
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Gestion des séries où le prix ne baisse jamais
    df.loc[avg_loss == 0, "rsi"] = 100

    # ATR sur 14 périodes
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ADX : mesure la force de la tendance
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    atr_safe = df["atr"].replace(0, np.nan)

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / 14, min_periods=14, adjust=False
        ).mean()
        / atr_safe
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / 14, min_periods=14, adjust=False
        ).mean()
        / atr_safe
    )

    di_sum = (plus_di + minus_di).replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / di_sum

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = dx.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    return df


# =========================================================
# 5C. STRUCTURE DU MARCHÉ
# =========================================================

def analyze_market_structure(data, lookback=30):
    """Évalue la structure récente avec les sommets et creux."""

    if data is None or len(data) < 10:
        return {
            "structure": "Données insuffisantes",
            "last_high": None,
            "last_low": None,
        }

    recent = data.tail(lookback).copy()

    last_high = float(recent["high"].max())
    last_low = float(recent["low"].min())
    last_close = float(recent["close"].iloc[-1])

    midpoint = (last_high + last_low) / 2

    if last_close > midpoint:
        structure = "Biais haussier"
    elif last_close < midpoint:
        structure = "Biais baissier"
    else:
        structure = "Neutre"

    return {
        "structure": structure,
        "last_high": last_high,
        "last_low": last_low,
    }


# =========================================================
# 5D. BALAYAGE DE LIQUIDITÉ
# =========================================================

def detect_liquidity_sweep(data, lookback=20):
    """
    Détecte une possible prise de liquidité :
    - Sweep haussier : le prix passe sous un ancien creux,
      puis clôture au-dessus de ce creux.
    - Sweep baissier : le prix dépasse un ancien sommet,
      puis clôture sous ce sommet.
    """

    result = {
        "sweep": "Aucun balayage détecté",
        "level": None,
    }

    if data is None or len(data) < lookback + 2:
        return result

    # On compare la dernière bougie aux bougies précédentes.
    previous = data.iloc[-(lookback + 1):-1]
    last = data.iloc[-1]

    previous_low = float(previous["low"].min())
    previous_high = float(previous["high"].max())

    if (
        float(last["low"]) < previous_low
        and float(last["close"]) > previous_low
    ):
        result["sweep"] = "Sweep haussier potentiel"
        result["level"] = previous_low

    elif (
        float(last["high"]) > previous_high
        and float(last["close"]) < previous_high
    ):
        result["sweep"] = "Sweep baissier potentiel"
        result["level"] = previous_high

    return result


# =========================================================
# 5E. ANALYSE D'UNE UNITÉ DE TEMPS
# =========================================================

def analyze_timeframe(raw_data):
    """Produit une synthèse technique pour une unité de temps."""

    if raw_data is None or len(raw_data) < 50:
        return {
            "status": "Données insuffisantes",
            "bias": "Neutre",
            "rsi": None,
            "adx": None,
            "atr": None,
            "macd": "Indisponible",
            "structure": "Indisponible",
            "sweep": "Indisponible",
            "close": None,
        }

    df = calculate_indicators(raw_data)
    df = df.dropna(subset=["ema_20", "ema_50", "rsi", "atr"])

    if df.empty:
        return {
            "status": "Indicateurs indisponibles",
            "bias": "Neutre",
            "rsi": None,
            "adx": None,
            "atr": None,
            "macd": "Indisponible",
            "structure": "Indisponible",
            "sweep": "Indisponible",
            "close": None,
        }

    # La dernière bougie peut encore être en formation.
    # On privilégie la dernière bougie clôturée.
    if len(df) >= 2:
        closed = df.iloc[:-1].copy()
    else:
        closed = df.copy()

    last = closed.iloc[-1]

    bullish_points = 0
    bearish_points = 0

    if last["close"] > last["ema_20"]:
        bullish_points += 1
    elif last["close"] < last["ema_20"]:
        bearish_points += 1

    if last["ema_20"] > last["ema_50"]:
        bullish_points += 1
    elif last["ema_20"] < last["ema_50"]:
        bearish_points += 1

    if last["macd"] > last["macd_signal"]:
        bullish_points += 1
    elif last["macd"] < last["macd_signal"]:
        bearish_points += 1

    if last["rsi"] > 55:
        bullish_points += 1
    elif last["rsi"] < 45:
        bearish_points += 1

    if bullish_points > bearish_points:
        bias = "Haussier"
    elif bearish_points > bullish_points:
        bias = "Baissier"
    else:
        bias = "Neutre"

    structure = analyze_market_structure(closed)
    sweep = detect_liquidity_sweep(closed)

    if last["macd"] > last["macd_signal"]:
        macd_state = "Haussier"
    elif last["macd"] < last["macd_signal"]:
        macd_state = "Baissier"
    else:
        macd_state = "Neutre"

    return {
        "status": "OK",
        "bias": bias,
        "rsi": float(last["rsi"]),
        "adx": (
            float(last["adx"])
            if pd.notna(last["adx"])
            else None
        ),
        "atr": float(last["atr"]),
        "macd": macd_state,
        "structure": structure["structure"],
        "sweep": sweep["sweep"],
        "close": float(last["close"]),
        "ema_20": float(last["ema_20"]),
        "ema_50": float(last["ema_50"]),
        "ema_200": (
            float(last["ema_200"])
            if pd.notna(last["ema_200"])
            else None
        ),
        "bullish_points": bullish_points,
        "bearish_points": bearish_points,
        "candle_time": last["time"],
    }


# =========================================================
# 5F. ANALYSE MULTI-UNITÉS DE TEMPS
# =========================================================

def analyze_multiple_timeframes(symbol, count=300):
    """Récupère et analyse les différentes unités de temps."""

    results = {}

    for tf in ["M5", "M15", "M30", "H1", "H4", "D1"]:
        raw_data = fetch_candles(symbol, tf, count)

        if raw_data.empty:
            results[tf] = {
                "status": "Données indisponibles",
                "bias": "Neutre",
                "rsi": None,
                "adx": None,
                "atr": None,
                "macd": "Indisponible",
                "structure": "Indisponible",
                "sweep": "Indisponible",
                "close": None,
            }
            continue

        results[tf] = analyze_timeframe(raw_data)

    return results
    
# =========================================================
# 5G. MOTEUR DE SIGNAUX FOREXSCOPE
# =========================================================

def generate_trading_signal(raw_data, multi_results):
    """
    Génère un signal indicatif à partir des indicateurs,
    de la structure et de la confluence multi-unités de temps.

    Le score est un score de confluence, pas une probabilité
    statistique de gain.
    """

    neutral_result = {
        "signal": "ATTENDRE",
        "confidence": 0,
        "entry": None,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward": None,
        "reasons": ["Données insuffisantes pour décider."],
    }

    if raw_data is None or len(raw_data) < 50:
        return neutral_result

    df = calculate_indicators(raw_data)
    df = df.dropna(subset=["ema_20", "ema_50", "rsi", "atr"])

    if len(df) < 2:
        return neutral_result

    # Utilise la dernière bougie clôturée.
    last = df.iloc[-2]

    close = float(last["close"])
    atr = float(last["atr"])
    rsi = float(last["rsi"])

    if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
        return neutral_result

    bullish_score = 0
    bearish_score = 0
    reasons = []

    # 1. Tendance locale : EMA
    if last["ema_20"] > last["ema_50"]:
        bullish_score += 2
        reasons.append("EMA 20 supérieure à EMA 50.")
    elif last["ema_20"] < last["ema_50"]:
        bearish_score += 2
        reasons.append("EMA 20 inférieure à EMA 50.")

    # 2. Position du prix
    if close > last["ema_20"]:
        bullish_score += 1
    elif close < last["ema_20"]:
        bearish_score += 1

    # 3. RSI : momentum, sans acheter automatiquement en survente
    if 50 < rsi < 70:
        bullish_score += 1
        reasons.append("RSI favorable au momentum haussier.")
    elif 30 < rsi < 50:
        bearish_score += 1
        reasons.append("RSI favorable au momentum baissier.")
    elif rsi >= 70:
        reasons.append("RSI élevé : risque de poursuite tardive.")
    elif rsi <= 30:
        reasons.append("RSI faible : risque de vente tardive.")

    # 4. MACD
    if last["macd"] > last["macd_signal"]:
        bullish_score += 1
    elif last["macd"] < last["macd_signal"]:
        bearish_score += 1

    # 5. Balayage de liquidité sur les bougies clôturées
    closed = df.iloc[:-1].copy()
    sweep = detect_liquidity_sweep(closed)

    if sweep["sweep"] == "Sweep haussier potentiel":
        bullish_score += 2
        reasons.append("Balayage haussier potentiel détecté.")
    elif sweep["sweep"] == "Sweep baissier potentiel":
        bearish_score += 2
        reasons.append("Balayage baissier potentiel détecté.")

    # 6. Confluence des unités supérieures.
    # On ne compte pas l'unité locale deux fois si elle
    # apparaît dans la liste multi-unités.
    higher_timeframes = ["M15", "M30", "H1", "H4", "D1"]

    for tf in higher_timeframes:
        result = multi_results.get(tf, {})

        if result.get("status") != "OK":
            continue

        bias = result.get("bias")

        if bias == "Haussier":
            bullish_score += 1
        elif bias == "Baissier":
            bearish_score += 1

    total_score = bullish_score + bearish_score

    if total_score == 0:
        return {
            **neutral_result,
            "reasons": ["Aucun élément directionnel suffisamment clair."],
        }

    # Une direction ne suffit pas : il faut une marge
    # minimale entre les scores et une confluence suffisante.
    score_difference = abs(bullish_score - bearish_score)

    if (
        bullish_score >= 5
        and bullish_score > bearish_score
        and score_difference >= 2
    ):
        signal = "ACHAT"
        winning_score = bullish_score
        direction = 1

    elif (
        bearish_score >= 5
        and bearish_score > bullish_score
        and score_difference >= 2
    ):
        signal = "VENTE"
        winning_score = bearish_score
        direction = -1

    else:
        signal = "ATTENDRE"
        winning_score = max(bullish_score, bearish_score)
        direction = 0
        reasons.append(
            "Confluence insuffisante ou signaux contradictoires."
        )

    # Score normalisé de confluence : ce n'est pas une
    # probabilité de réussite.
    confidence = round(
        100 * winning_score / max(total_score, 1)
    )

    entry = close
    stop_loss = None
    take_profit = None
    risk_reward = None

    if direction != 0:
        # Distance de protection fondée sur l'ATR.
        risk_distance = 1.5 * atr
        reward_distance = 3.0 * atr

        if direction == 1:
            stop_loss = entry - risk_distance
            take_profit = entry + reward_distance
        else:
            stop_loss = entry + risk_distance
            take_profit = entry - reward_distance

        risk_reward = reward_distance / risk_distance

    return {
        "signal": signal,
        "confidence": confidence,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": risk_reward,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "reasons": reasons,
        "rsi": rsi,
        "atr": atr,
    }
    
    
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
    st.caption(
        "Les résultats reposent sur les bougies clôturées "
        "et les indicateurs calculés. Ils ne garantissent pas "
        "la direction future du marché."
    )

    if candles.empty:
        st.warning(
            "Les données du marché sélectionné sont indisponibles. "
            "Impossible de produire une analyse fiable."
        )
    else:
        # Analyse de l'unité de temps actuellement sélectionnée
        current_analysis = analyze_timeframe(candles)

        st.markdown("### Synthèse du marché sélectionné")
        st.write(f"**Marché :** {market_name}")
        st.write(f"**Unité de temps :** {timeframe}")

        if current_analysis["status"] != "OK":
            st.warning(current_analysis["status"])
        else:
            a, b, c = st.columns(3)

            a.metric(
                "Biais technique",
                current_analysis["bias"],
            )

            b.metric(
                "RSI (14)",
                f"{current_analysis['rsi']:.2f}",
            )

            adx_value = current_analysis["adx"]
            c.metric(
                "ADX (14)",
                f"{adx_value:.2f}"
                if adx_value is not None
                else "Indisponible",
            )

            d, e, f = st.columns(3)

            d.metric(
                "Dernière clôture analysée",
                format_price(current_analysis["close"]),
            )

            e.metric(
                "MACD",
                current_analysis["macd"],
            )

            f.metric(
                "Structure du marché",
                current_analysis["structure"],
            )

            st.write(
                "**Balayage de liquidité :** "
                + current_analysis["sweep"]
            )

            st.write(
                "**ATR (14) :** "
                + format_price(current_analysis["atr"])
            )

            st.divider()

        # Analyse de plusieurs unités de temps
        st.markdown("### Analyse multi-unités de temps")

        if st.button(
            "Lancer l'analyse multi-unités",
            key="run_multi_analysis",
            use_container_width=True,
        ):
            with st.spinner(
                "Récupération et analyse des différentes unités..."
            ):
                multi_results = analyze_multiple_timeframes(
                    SYMBOLS[market_name],
                    candle_count,
                )

            rows = []

            for tf, result in multi_results.items():
                rows.append({
                    "Unité": tf,
                    "État": result.get("status", "Inconnu"),
                    "Biais": result.get("bias", "Neutre"),
                    "RSI": (
                        round(result["rsi"], 2)
                        if result.get("rsi") is not None
                        else None
                    ),
                    "ADX": (
                        round(result["adx"], 2)
                        if result.get("adx") is not None
                        else None
                    ),
                    "MACD": result.get("macd", "Indisponible"),
                    "Structure": result.get(
                        "structure", "Indisponible"
                    ),
                    "Liquidité": result.get(
                        "sweep", "Indisponible"
                    ),
                })

            result_df = pd.DataFrame(rows)

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
            )

            valid_results = [
                result
                for result in multi_results.values()
                if result.get("status") == "OK"
            ]

            if valid_results:
                bullish = sum(
                    r["bias"] == "Haussier"
                    for r in valid_results
                )

                bearish = sum(
                    r["bias"] == "Baissier"
                    for r in valid_results
                )

                neutral = sum(
                    r["bias"] == "Neutre"
                    for r in valid_results
                )

                st.markdown("### Répartition des biais")

                x, y, z = st.columns(3)

                x.metric("Unités haussières", bullish)
                y.metric("Unités baissières", bearish)
                z.metric("Unités neutres", neutral)

                if bullish > bearish:
                    st.info(
                        "Les biais haussiers sont majoritaires "
                        "parmi les unités analysées disponibles."
                    )
                elif bearish > bullish:
                    st.info(
                        "Les biais baissiers sont majoritaires "
                        "parmi les unités analysées disponibles."
                    )
                else:
                    st.info(
                        "Les biais haussiers et baissiers sont "
                        "à égalité parmi les unités analysées."
                    )

            else:
                st.warning(
                    "Aucune unité de temps n'a fourni assez de "
                    "données valides pour une analyse."
)
 
            st.divider()
            st.markdown("### Signal ForexScope")

            signal_result = generate_trading_signal(
                candles,
                multi_results,
            )

            s1, s2, s3 = st.columns(3)

            s1.metric("Décision", signal_result["signal"])
            s2.metric(
                "Score de confluence",
                f"{signal_result['confidence']} / 100",
            )

            s3.metric(
                "Ratio risque/rendement",
                (
                    f"1:{signal_result['risk_reward']:.1f}"
                    if signal_result["risk_reward"] is not None
                    else "Non défini"
                ),
            )

            if signal_result["signal"] == "ACHAT":
                st.success("Biais haussier détecté. Vérifie le contexte avant toute entrée.")
            elif signal_result["signal"] == "VENTE":
                st.error("Biais baissier détecté. Vérifie le contexte avant toute entrée.")
            else:
                st.warning("Pas de confluence suffisante : aucune entrée proposée.")

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Entrée indicative",
                (
                    format_price(signal_result["entry"])
                    if signal_result["entry"] is not None
                    else "—"
                ),
            )

            p2.metric(
                "Stop Loss",
                (
                    format_price(signal_result["stop_loss"])
                    if signal_result["stop_loss"] is not None
                    else "—"
                ),
            )

            p3.metric(
                "Take Profit",
                (
                    format_price(signal_result["take_profit"])
                    if signal_result["take_profit"] is not None
                    else "—"
                ),
            )

            with st.expander("Pourquoi ce résultat ?"):
                for reason in signal_result["reasons"]:
                    st.write(f"- {reason}")
        
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

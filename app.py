
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import sqlite3

from datetime import datetime, timezone


# =========================================================
# 1. CONFIGURATION DE L'APPLICATION
# =========================================================

st.set_page_config(
    page_title="ForexScope",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    [data-testid="stSidebar"] {
        background-color: #151a23;
    }

    h1, h2, h3 {
        color: #e6edf3;
    }

    div[data-testid="stMetric"] {
        background-color: #171d28;
        border: 1px solid #293241;
        padding: 12px;
        border-radius: 10px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. MARCHÉS DISPONIBLES
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


# =========================================================
# 3. UNITÉS DE TEMPS
# =========================================================

INTERVALS = {
    "M1": {
        "interval": "1m",
        "period": "7d"
    },
    "M5": {
        "interval": "5m",
        "period": "60d"
    },
    "M15": {
        "interval": "15m",
        "period": "60d"
    },
    "M30": {
        "interval": "30m",
        "period": "60d"
    },
    "H1": {
        "interval": "60m",
        "period": "60d"
    },
    "H4": {
        "interval": "60m",
        "period": "60d"
    },
    "D1": {
        "interval": "1d",
        "period": "2y"
    },
}


# =========================================================
# 4. NETTOYAGE DES DONNÉES OHLC
# =========================================================

def validate_ohlc(data):
    """
    Nettoie les données de marché et conserve :
    time, open, high, low, close.
    """

    if data is None or data.empty:
        return pd.DataFrame(
            columns=["time", "open", "high", "low", "close"]
        )

    df = data.copy()

    # Yahoo Finance peut renvoyer des colonnes MultiIndex.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            str(col[0]).lower()
            for col in df.columns
        ]
    else:
        df.columns = [
            str(col).lower()
            for col in df.columns
        ]

    df = df.loc[:, ~df.columns.duplicated()]

    required = ["open", "high", "low", "close"]

    for column in required:
        if column not in df.columns:
            return pd.DataFrame(
                columns=["time", "open", "high", "low", "close"]
            )

    # L'index Yahoo Finance correspond généralement à la date.
    df["time"] = pd.to_datetime(
        df.index,
        utc=True,
        errors="coerce"
    )

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df[
        ["time", "open", "high", "low", "close"]
    ].copy()

    df = df.dropna()

    # Éliminer les prix invalides.
    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
    ]

    # Vérification de la cohérence des bougies.
    df = df[
        (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    ]

    df = df.drop_duplicates(
        subset=["time"],
        keep="last"
    )

    df = df.sort_values("time").reset_index(drop=True)

    return df


# =========================================================
# 5. RÉCUPÉRATION DES BOUGIES
# =========================================================

@st.cache_data(ttl=30, show_spinner=False)
def fetch_candles(market_name, timeframe, candle_count=300):
    """
    Récupère les bougies depuis Yahoo Finance.

    Attention :
    - Yahoo Finance n'est pas un flux de cotations garanti en temps réel.
    - L'historique intraday disponible dépend du marché et de Yahoo.
    - H4 est reconstruit à partir des bougies H1.
    """

    if market_name not in SYMBOLS:
        return pd.DataFrame()

    if timeframe not in INTERVALS:
        return pd.DataFrame()

    symbol = SYMBOLS[market_name]
    settings = INTERVALS[timeframe]

    # H4 est construit à partir des données H1.
    if timeframe == "H4":
        interval = "60m"
        period = "60d"
    else:
        interval = settings["interval"]
        period = settings["period"]

    try:
        raw = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False
        )

        df = validate_ohlc(raw)

        if df.empty:
            return df

        # Reconstruction des bougies de quatre heures.
        if timeframe == "H4":
            df = df.set_index("time")

            df = df.resample(
                "4h",
                origin="start_day",
                label="left",
                closed="left"
            ).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last"
            })

            df = df.dropna().reset_index()

            df = validate_ohlc(df)

        if df.empty:
            return df

        # Le nombre de bougies retournées est limité.
        return df.tail(int(candle_count)).reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


# =========================================================
# 6. CALCUL DES INDICATEURS TECHNIQUES
# =========================================================

def calculate_indicators(data):
    """
    Calcule les indicateurs nécessaires à l'analyse :
    EMA 20/50/200, RSI, ATR, MACD, ADX et DI.
    """

    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()

    for column in ["open", "high", "low", "close"]:
        if column not in df.columns:
            return pd.DataFrame()

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=["open", "high", "low", "close"]
    ).reset_index(drop=True)

    if df.empty:
        return df

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # -------------------------
    # Moyennes mobiles EMA
    # -------------------------

    df["EMA20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA200"] = close.ewm(
        span=200,
        adjust=False
    ).mean()

    # -------------------------
    # RSI sur 14 périodes
    # -------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # Cas particuliers pour éviter les valeurs manquantes
    # lorsque les prix ne montent ou ne baissent jamais.
    df.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI"
    ] = 100

    df.loc[
        (avg_gain == 0) & (avg_loss > 0),
        "RSI"
    ] = 0

    df.loc[
        (avg_gain == 0) & (avg_loss == 0),
        "RSI"
    ] = 50

    # -------------------------
    # ATR sur 14 périodes
    # -------------------------

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    # -------------------------
    # MACD
    # -------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_hist"] = (
        df["MACD"] - df["MACD_signal"]
    )

    # -------------------------
    # ADX et indicateurs directionnels
    # -------------------------

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    atr_safe = df["ATR"].replace(0, np.nan)

    df["PLUS_DI"] = (
        100
        * plus_dm.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()
        / atr_safe
    )

    df["MINUS_DI"] = (
        100
        * minus_dm.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()
        / atr_safe
    )

    di_sum = (
        df["PLUS_DI"] + df["MINUS_DI"]
    ).replace(0, np.nan)

    df["DX"] = (
        100
        * (
            df["PLUS_DI"] - df["MINUS_DI"]
        ).abs()
        / di_sum
    )

    df["ADX"] = df["DX"].ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    return df

# =========================================================
# 7. ANALYSE DE LA STRUCTURE DU MARCHÉ
# =========================================================

def analyze_market_structure(data, lookback=20):
    """
    Analyse simplifiée de la structure :
    - sommets et creux récents ;
    - cassure haussière ou baissière ;
    - orientation de la structure.
    """

    result = {
        "bias": "NEUTRE",
        "structure": "INDÉTERMINÉE",
        "last_high": None,
        "last_low": None,
        "breakout": None
    }

    if data is None or len(data) < max(lookback + 2, 10):
        return result

    df = data.copy()
    closed = df.iloc[:-1].copy()

    if len(closed) < lookback + 1:
        return result

    current = closed.iloc[-1]
    previous = closed.iloc[-lookback - 1:-1]

    if previous.empty:
        return result

    previous_high = float(previous["high"].max())
    previous_low = float(previous["low"].min())
    close = float(current["close"])

    result["last_high"] = previous_high
    result["last_low"] = previous_low

    if close > previous_high:
        result["bias"] = "HAUSSIER"
        result["structure"] = "CASSURE HAUSSIÈRE"
        result["breakout"] = "BULLISH"

    elif close < previous_low:
        result["bias"] = "BAISSIER"
        result["structure"] = "CASSURE BAISSIÈRE"
        result["breakout"] = "BEARISH"

    else:
        recent = closed.tail(lookback)

        first_close = float(recent.iloc[0]["close"])
        last_close = float(recent.iloc[-1]["close"])

        if last_close > first_close:
            result["bias"] = "HAUSSIER"
            result["structure"] = "PROGRESSION"

        elif last_close < first_close:
            result["bias"] = "BAISSIER"
            result["structure"] = "RÉGRESSION"

        else:
            result["bias"] = "NEUTRE"
            result["structure"] = "RANGE"

    return result


# =========================================================
# 8. DÉTECTION DES BALAYAGES DE LIQUIDITÉ
# =========================================================

def detect_liquidity_sweep(data, lookback=20):
    """
    Détecte un balayage simplifié :
    - sweep haussier : le prix passe sous un ancien creux
      puis clôture au-dessus de ce creux ;
    - sweep baissier : le prix dépasse un ancien sommet
      puis clôture sous ce sommet.

    C'est un indice potentiel, pas une preuve de mouvement futur.
    """

    result = {
        "type": "AUCUN",
        "level": None,
        "description": "Aucun balayage détecté"
    }

    if data is None or len(data) < lookback + 3:
        return result

    df = data.copy()

    # La dernière bougie peut être encore en formation.
    closed = df.iloc[:-1].copy()

    if len(closed) < lookback + 1:
        return result

    candle = closed.iloc[-1]
    previous = closed.iloc[-lookback - 1:-1]

    if previous.empty:
        return result

    previous_high = float(previous["high"].max())
    previous_low = float(previous["low"].min())

    candle_high = float(candle["high"])
    candle_low = float(candle["low"])
    candle_close = float(candle["close"])

    # Balayage sous les creux, puis clôture au-dessus.
    if candle_low < previous_low and candle_close > previous_low:
        result = {
            "type": "SWEEP HAUSSIER",
            "level": previous_low,
            "description": (
                "Le prix a dépassé un ancien creux "
                "puis clôturé au-dessus."
            )
        }

    # Balayage au-dessus des sommets, puis clôture en dessous.
    elif candle_high > previous_high and candle_close < previous_high:
        result = {
            "type": "SWEEP BAISSIER",
            "level": previous_high,
            "description": (
                "Le prix a dépassé un ancien sommet "
                "puis clôturé en dessous."
            )
        }

    return result


# =========================================================
# 9. ANALYSE D'UNE UNITÉ DE TEMPS
# =========================================================

def analyze_timeframe(data, timeframe):
    """
    Analyse une unité de temps avec les indicateurs
    et la structure du marché.
    """

    result = {
        "timeframe": timeframe,
        "bias": "NEUTRE",
        "structure": "INDÉTERMINÉE",
        "close": None,
        "ema20": None,
        "ema50": None,
        "ema200": None,
        "rsi": None,
        "atr": None,
        "adx": None,
        "macd": None,
        "macd_signal": None,
        "liquidity_sweep": "AUCUN",
        "liquidity_level": None,
        "candle_time": None,
        "reason": "Données insuffisantes"
    }

    if data is None or len(data) < 30:
        return result

    df = calculate_indicators(data)

    if df.empty or len(df) < 30:
        return result

    # Utilise la dernière bougie clôturée.
    candle = df.iloc[-2]

    numeric_fields = [
        "close", "EMA20", "EMA50", "EMA200",
        "RSI", "ATR", "ADX", "MACD", "MACD_signal"
    ]

    for field in numeric_fields:
        if field not in df.columns or pd.isna(candle[field]):
            return result

    close = float(candle["close"])
    ema20 = float(candle["EMA20"])
    ema50 = float(candle["EMA50"])
    ema200 = float(candle["EMA200"])
    rsi = float(candle["RSI"])
    atr = float(candle["ATR"])
    adx = float(candle["ADX"])
    macd = float(candle["MACD"])
    macd_signal = float(candle["MACD_signal"])

    structure = analyze_market_structure(df)
    sweep = detect_liquidity_sweep(df)

    bullish_points = 0
    bearish_points = 0

    # Position du prix par rapport aux EMA.
    if close > ema20:
        bullish_points += 1
    elif close < ema20:
        bearish_points += 1

    if ema20 > ema50:
        bullish_points += 1
    elif ema20 < ema50:
        bearish_points += 1

    if close > ema200:
        bullish_points += 1
    elif close < ema200:
        bearish_points += 1

    # RSI : on évite de considérer un RSI très élevé
    # comme une confirmation automatique d'achat.
    if 50 < rsi < 70:
        bullish_points += 1
    elif 30 < rsi < 50:
        bearish_points += 1

    # MACD.
    if macd > macd_signal:
        bullish_points += 1
    elif macd < macd_signal:
        bearish_points += 1

    # Structure du marché.
    if structure["bias"] == "HAUSSIER":
        bullish_points += 1
    elif structure["bias"] == "BAISSIER":
        bearish_points += 1

    # Balayage de liquidité.
    if sweep["type"] == "SWEEP HAUSSIER":
        bullish_points += 1
    elif sweep["type"] == "SWEEP BAISSIER":
        bearish_points += 1

    if bullish_points >= bearish_points + 2:
        bias = "HAUSSIER"
    elif bearish_points >= bullish_points + 2:
        bias = "BAISSIER"
    else:
        bias = "NEUTRE"

    try:
        candle_time = pd.Timestamp(candle["time"]).isoformat()
    except Exception:
        candle_time = None

    result.update({
        "bias": bias,
        "structure": structure["structure"],
        "close": close,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "atr": atr,
        "adx": adx,
        "macd": macd,
        "macd_signal": macd_signal,
        "liquidity_sweep": sweep["type"],
        "liquidity_level": sweep["level"],
        "candle_time": candle_time,
        "reason": (
            f"Points haussiers : {bullish_points}; "
            f"points baissiers : {bearish_points}"
        )
    })

    return result


# =========================================================
# 10. ANALYSE MULTI-UNITÉS DE TEMPS
# =========================================================

def analyze_multiple_timeframes(market_name, candle_count=300):
    """
    Analyse plusieurs unités de temps.
    Les résultats servent à mesurer la confluence.

    Les unités de temps ne sont pas toutes disponibles
    avec la même profondeur de données chez Yahoo Finance.
    """

    timeframes = ["M5", "M15", "M30", "H1", "H4", "D1"]

    results = {}

    for timeframe in timeframes:
        candles = fetch_candles(
            market_name,
            timeframe,
            candle_count
        )

        results[timeframe] = analyze_timeframe(
            candles,
            timeframe
        )

    return results


# =========================================================
# 11. GÉNÉRATION DU SIGNAL ET DES NIVEAUX SL/TP
# =========================================================

def generate_trading_signal(raw_data, multi_results):
    """
    Génère un signal indicatif :
    ACHAT, VENTE ou ATTENDRE.

    Le SL et le TP sont calculés à partir de l'ATR.
    Le signal n'est pas une garantie de rentabilité.
    """

    neutral_result = {
        "signal": "ATTENDRE",
        "confidence": 0,
        "entry": None,
        "sl": None,
        "tp": None,
        "risk_reward": None,
        "reason": "Données insuffisantes pour générer un signal.",
        "candle_time": None,
        "created_at_utc": datetime.now(timezone.utc).isoformat()
    }

    if raw_data is None or len(raw_data) < 30:
        return neutral_result

    df = calculate_indicators(raw_data)

    if df.empty or len(df) < 30:
        return neutral_result

    # Analyse uniquement sur une bougie clôturée.
    last = df.iloc[-2]

    required = [
        "close", "EMA20", "EMA50", "EMA200",
        "RSI", "ATR", "MACD", "MACD_signal"
    ]

    if any(
        column not in df.columns or pd.isna(last[column])
        for column in required
    ):
        return neutral_result

    entry = float(last["close"])
    atr = float(last["ATR"])
    rsi = float(last["RSI"])

    if not np.isfinite(entry) or not np.isfinite(atr):
        return neutral_result

    if entry <= 0 or atr <= 0:
        return neutral_result

    bullish_score = 0
    bearish_score = 0
    reasons = []

    # -------------------------
    # Tendance EMA
    # -------------------------

    if last["EMA20"] > last["EMA50"]:
        bullish_score += 1
        reasons.append("EMA20 au-dessus de EMA50")

    elif last["EMA20"] < last["EMA50"]:
        bearish_score += 1
        reasons.append("EMA20 sous EMA50")

    if entry > last["EMA200"]:
        bullish_score += 1
        reasons.append("Prix au-dessus de EMA200")

    elif entry < last["EMA200"]:
        bearish_score += 1
        reasons.append("Prix sous EMA200")

    # -------------------------
    # RSI
    # -------------------------

    if 50 < rsi < 70:
        bullish_score += 1
        reasons.append("RSI favorable aux acheteurs")

    elif 30 < rsi < 50:
        bearish_score += 1
        reasons.append("RSI favorable aux vendeurs")

    elif rsi >= 70:
        reasons.append("RSI élevé : risque de surachat")

    elif rsi <= 30:
        reasons.append("RSI bas : risque de survente")

    # -------------------------
    # MACD
    # -------------------------

    if last["MACD"] > last["MACD_signal"]:
        bullish_score += 1
        reasons.append("MACD haussier")

    elif last["MACD"] < last["MACD_signal"]:
        bearish_score += 1
        reasons.append("MACD baissier")

    # -------------------------
    # Confluence multi-unités
    # -------------------------

    relevant_timeframes = ["M15", "M30", "H1", "H4", "D1"]

    bullish_tf = 0
    bearish_tf = 0
    usable_tf = 0

    for timeframe in relevant_timeframes:
        item = multi_results.get(timeframe, {})

        if item.get("bias") == "HAUSSIER":
            bullish_tf += 1
            usable_tf += 1

        elif item.get("bias") == "BAISSIER":
            bearish_tf += 1
            usable_tf += 1

    # La confluence pèse dans la décision, sans remplacer
    # les indicateurs de l'unité de temps d'entrée.
    if bullish_tf >= 3:
        bullish_score += 2
        reasons.append(
            f"Confluence haussière sur {bullish_tf} unités"
        )

    elif bearish_tf >= 3:
        bearish_score += 2
        reasons.append(
            f"Confluence baissière sur {bearish_tf} unités"
        )

    # -------------------------
    # Balayage de liquidité
    # -------------------------

    sweep = detect_liquidity_sweep(df)

    if sweep["type"] == "SWEEP HAUSSIER":
        bullish_score += 1
        reasons.append("Balayage haussier de liquidité")

    elif sweep["type"] == "SWEEP BAISSIER":
        bearish_score += 1
        reasons.append("Balayage baissier de liquidité")

    # -------------------------
    # Décision finale
    # -------------------------

    score_difference = bullish_score - bearish_score

    if score_difference >= 3:
        signal = "ACHAT"
        direction = 1

    elif score_difference <= -3:
        signal = "VENTE"
        direction = -1

    else:
        signal = "ATTENDRE"
        direction = 0

    total_score = bullish_score + bearish_score

    if total_score > 0:
        confidence = int(
            round(
                100
                * max(bullish_score, bearish_score)
                / total_score
            )
        )
    else:
        confidence = 0

    # La confiance reflète uniquement la proportion des scores.
    # Ce n'est pas une probabilité statistique de réussite.

    sl = None
    tp = None
    risk_reward = None

    if direction == 1:
        sl = entry - (1.5 * atr)
        tp = entry + (3.0 * atr)
        risk_reward = 2.0

    elif direction == -1:
        sl = entry + (1.5 * atr)
        tp = entry - (3.0 * atr)
        risk_reward = 2.0

    try:
        candle_time = pd.Timestamp(last["time"]).isoformat()
    except Exception:
        candle_time = None

    return {
        "signal": signal,
        "confidence": confidence,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk_reward": risk_reward,
        "reason": " | ".join(reasons) if reasons else (
            "Pas de confluence suffisante."
        ),
        "candle_time": candle_time,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "bullish_timeframes": bullish_tf,
        "bearish_timeframes": bearish_tf,
        "usable_timeframes": usable_tf
    }


# =========================================================
# 12. BASE DE DONNÉES SQLITE
# =========================================================

DB_PATH = "forexscope.db"


def init_signal_db():
    """Crée la table d'historique si elle n'existe pas."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                viewed_at_utc TEXT,
                candle_time_utc TEXT,
                market TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence INTEGER,
                entry REAL,
                stop_loss REAL,
                take_profit REAL,
                risk_reward REAL,
                reason TEXT
            )
        """)

        # Mise à niveau des anciennes bases créées
        # avec une version plus simple de l'application.
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(signal_history)"
            ).fetchall()
        }

        if "candle_time_utc" not in columns:
            conn.execute(
                "ALTER TABLE signal_history "
                "ADD COLUMN candle_time_utc TEXT"
            )

        conn.commit()


def save_signal_record(market, timeframe, result):
    """Enregistre un signal une seule fois lors de l'analyse."""

    init_signal_db()

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            INSERT INTO signal_history (
                created_at_utc,
                candle_time_utc,
                market,
                timeframe,
                signal,
                confidence,
                entry,
                stop_loss,
                take_profit,
                risk_reward,
                reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.get(
                "created_at_utc",
                datetime.now(timezone.utc).isoformat()
            ),
            result.get("candle_time"),
            market,
            timeframe,
            result.get("signal"),
            result.get("confidence"),
            result.get("entry"),
            result.get("sl"),
            result.get("tp"),
            result.get("risk_reward"),
            result.get("reason")
        ))

        conn.commit()

        return cursor.lastrowid


def get_signal_history(limit=100):
    """Retourne les signaux les plus récents."""

    init_signal_db()

    with sqlite3.connect(DB_PATH) as conn:
        history = pd.read_sql_query("""
            SELECT
                id,
                created_at_utc,
                candle_time_utc,
                market,
                timeframe,
                signal,
                confidence,
                entry,
                stop_loss,
                take_profit,
                risk_reward,
                reason
            FROM signal_history
            ORDER BY id DESC
            LIMIT ?
        """, conn, params=(int(limit),))

    return history

def mark_signals_as_viewed(signal_ids):
    """Enregistre la première consultation des signaux."""

    if not signal_ids:
        return

    init_signal_db()

    viewed_at = datetime.now(timezone.utc).isoformat()

    placeholders = ",".join(
        ["?"] * len(signal_ids)
    )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"""
            UPDATE signal_history
            SET viewed_at_utc = ?
            WHERE id IN ({placeholders})
            AND viewed_at_utc IS NULL
            """,
            [viewed_at] + list(signal_ids)
        )
        conn.commit()


# Initialiser la base de données.
init_signal_db()

# =========================================================
# 13. FONCTIONS D'AFFICHAGE
# =========================================================

def format_price(value, market_name):
    """Formate un prix selon le marché."""

    if value is None or pd.isna(value):
        return "—"

    try:
        price = float(value)

        if "JPY" in market_name:
            return f"{price:.3f}"

        if market_name in ["Or (Gold)", "Argent (Silver)"]:
            return f"{price:.2f}"

        if market_name == "Bitcoin / USD":
            return f"{price:,.2f}"

        if market_name in ["S&P 500", "Nasdaq 100"]:
            return f"{price:,.2f}"

        return f"{price:.5f}"

    except (ValueError, TypeError):
        return "—"


def build_candlestick_chart(data, market_name, timeframe):
    """Construit le graphique en chandeliers."""

    if data is None or data.empty:
        return None

    df = calculate_indicators(data)

    if df.empty:
        return None

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Prix"
    ))

    for column, color in [
        ("EMA20", "orange"),
        ("EMA50", "deepskyblue"),
        ("EMA200", "magenta")
    ]:
        if column in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time"],
                y=df[column],
                mode="lines",
                name=column,
                line=dict(width=1.2, color=color)
            ))

    fig.update_layout(
        title=f"{market_name} — {timeframe}",
        template="plotly_dark",
        height=580,
        xaxis_title="Heure",
        yaxis_title="Prix",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        margin=dict(l=10, r=10, t=70, b=10)
    )

    return fig


def utc_to_kinshasa(value):
    """Convertit une date UTC en heure de Kinshasa."""

    if value is None or pd.isna(value):
        return "—"

    try:
        timestamp = pd.to_datetime(value, utc=True)
        return timestamp.tz_convert(
            "Africa/Kinshasa"
        ).strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        return str(value)


# =========================================================
# 14. ÉTAT DE SESSION
# =========================================================

if "multi_results" not in st.session_state:
    st.session_state.multi_results = None

if "signal_result" not in st.session_state:
    st.session_state.signal_result = None

if "signal_id" not in st.session_state:
    st.session_state.signal_id = None

if "analysis_market" not in st.session_state:
    st.session_state.analysis_market = None

if "analysis_timeframe" not in st.session_state:
    st.session_state.analysis_timeframe = None

if "analysis_candle_count" not in st.session_state:
    st.session_state.analysis_candle_count = None


# =========================================================
# 15. BARRE LATÉRALE
# =========================================================

st.sidebar.title("ForexScope")
st.sidebar.caption("Analyse technique multi-unités")

market_name = st.sidebar.selectbox(
    "Marché",
    options=list(SYMBOLS.keys()),
    index=0
)

timeframe = st.sidebar.selectbox(
    "Unité de temps principale",
    options=list(INTERVALS.keys()),
    index=2
)

candle_count = st.sidebar.slider(
    "Nombre de bougies",
    min_value=100,
    max_value=1000,
    value=300,
    step=50
)

if st.sidebar.button(
    "Actualiser les données",
    use_container_width=True
):
    fetch_candles.clear()
    st.rerun()

st.sidebar.divider()

st.sidebar.caption("Indicateurs actifs")
st.sidebar.write("EMA 20 / 50 / 200")
st.sidebar.write("RSI 14")
st.sidebar.write("ATR 14")
st.sidebar.write("MACD")
st.sidebar.write("ADX et DI")
st.sidebar.write("Structure et liquidité")

st.sidebar.divider()

st.sidebar.caption(
    "Source des cotations : Yahoo Finance. "
    "Les données peuvent être retardées."
)


# =========================================================
# 16. TITRE ET RÉCUPÉRATION DES DONNÉES
# =========================================================

st.title("ForexScope")
st.caption(
    "Analyse technique, confluence multi-unités "
    "et gestion du risque."
)

candles = fetch_candles(
    market_name,
    timeframe,
    candle_count
)

if candles.empty:
    st.error(
        "Impossible de récupérer les bougies pour ce marché "
        "et cette unité de temps. Essaie une autre unité "
        "ou réessaie plus tard."
    )

    st.stop()
st.subheader("Connexion Yahoo Finance")

if candles.empty:
    st.error("Aucune donnée reçue")
else:
    st.success("Données reçues")

    last_time = pd.to_datetime(
        candles["time"].iloc[-1],
        utc=True
    )

    st.write(
        "Dernière bougie reçue (heure de Kinshasa) :",
        last_time.tz_convert(
            "Africa/Kinshasa"
        ).strftime("%d/%m/%Y %H:%M:%S")
    )

    st.write(
        "Heure actuelle (Kinshasa) :",
        datetime.now(
            timezone.utc
        ).astimezone(
            __import__("zoneinfo").ZoneInfo("Africa/Kinshasa")
        ).strftime("%d/%m/%Y %H:%M:%S")
    )
latest_closed = candles.iloc[-2] if len(candles) >= 2 else candles.iloc[-1]

price_now = float(latest_closed["close"])

col_market, col_timeframe, col_price, col_candles = st.columns(4)

with col_market:
    st.metric("Marché", market_name)

with col_timeframe:
    st.metric("Unité principale", timeframe)

with col_price:
    st.metric(
        "Dernière clôture",
        format_price(price_now, market_name)
    )

with col_candles:
    st.metric("Bougies chargées", len(candles))


# =========================================================
# 17. ONGLETS
# =========================================================

tab_market, tab_analysis, tab_history, tab_settings = st.tabs([
    "Marché",
    "Analyse",
    "Historique",
    "Paramètres"
])


# =========================================================
# 18. ONGLET MARCHÉ
# =========================================================

with tab_market:
    st.subheader("Graphique du marché")

    chart = build_candlestick_chart(
        candles,
        market_name,
        timeframe
    )

    if chart is not None:
        st.plotly_chart(
            chart,
            use_container_width=True
        )
    else:
        st.warning("Le graphique ne peut pas être construit.")

    st.subheader("Dernières bougies")

    recent = candles.tail(15).copy()

    recent["time"] = pd.to_datetime(
        recent["time"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert("Africa/Kinshasa").dt.strftime(
        "%d/%m/%Y %H:%M"
    )

    recent = recent.rename(columns={
        "time": "Heure (Kinshasa)",
        "open": "Ouverture",
        "high": "Plus haut",
        "low": "Plus bas",
        "close": "Clôture"
    })

    st.dataframe(
        recent.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Les données du tableau sont horodatées en heure de Kinshasa."
    )


# =========================================================
# 19. ONGLET ANALYSE
# =========================================================

with tab_analysis:
    st.subheader("Analyse de l'unité principale")

    indicator_data = calculate_indicators(candles)

    if len(indicator_data) >= 2:
        closed_candle = indicator_data.iloc[-2]

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.metric(
                "RSI 14",
                (
                    f"{closed_candle['RSI']:.2f}"
                    if pd.notna(closed_candle["RSI"])
                    else "—"
                )
            )

        with metric2:
            st.metric(
                "ATR 14",
                format_price(
                    closed_candle["ATR"],
                    market_name
                )
            )

        with metric3:
            st.metric(
                "ADX",
                (
                    f"{closed_candle['ADX']:.2f}"
                    if pd.notna(closed_candle["ADX"])
                    else "—"
                )
            )

        with metric4:
            if closed_candle["EMA20"] > closed_candle["EMA50"]:
                ema_bias = "Haussier"
            elif closed_candle["EMA20"] < closed_candle["EMA50"]:
                ema_bias = "Baissier"
            else:
                ema_bias = "Neutre"

            st.metric("EMA 20 / 50", ema_bias)

    st.divider()

    st.subheader("Analyse multi-unités")

    st.write(
        "L'analyse utilise plusieurs unités de temps pour "
        "chercher des confirmations. Les unités supérieures "
        "aident à contextualiser l'unité d'entrée."
    )

    run_analysis = st.button(
        "Lancer l'analyse multi-unités",
        type="primary",
        use_container_width=True
    )

    if run_analysis:
        with st.spinner(
            "Récupération des données et analyse des unités de temps..."
        ):
            multi_results = analyze_multiple_timeframes(
                market_name,
                candle_count
            )

            signal_result = generate_trading_signal(
                candles,
                multi_results
            )

            # Sauvegarde des résultats dans la session.
            st.session_state.multi_results = multi_results
            st.session_state.signal_result = signal_result

            st.session_state.analysis_market = market_name
            st.session_state.analysis_timeframe = timeframe
            st.session_state.analysis_candle_count = candle_count

            # Sauvegarder une seule fois au moment de l'analyse.
            st.session_state.signal_id = save_signal_record(
                market_name,
                timeframe,
                signal_result
            )

    results_are_current = (
        st.session_state.analysis_market == market_name
        and st.session_state.analysis_timeframe == timeframe
        and st.session_state.analysis_candle_count == candle_count
    )

    if st.session_state.multi_results is not None:
        if not results_are_current:
            st.warning(
                "Les résultats affichés correspondent à une "
                "analyse précédente. Relance l'analyse pour "
                "le marché, l'unité ou le nombre de bougies actuels."
            )

        multi_results = st.session_state.multi_results

        rows = []

        for tf, item in multi_results.items():
            rows.append({
                "Unité": tf,
                "Biais": item.get("bias", "NEUTRE"),
                "Structure": item.get("structure", "—"),
                "RSI": (
                    round(item["rsi"], 2)
                    if item.get("rsi") is not None
                    else None
                ),
                "ADX": (
                    round(item["adx"], 2)
                    if item.get("adx") is not None
                    else None
                ),
                "Balayage": item.get(
                    "liquidity_sweep",
                    "AUCUN"
                ),
                "Clôture analysée": utc_to_kinshasa(
                    item.get("candle_time")
                )
            })

        st.subheader("Résultats par unité de temps")

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

        bullish_count = sum(
            1 for item in multi_results.values()
            if item.get("bias") == "HAUSSIER"
        )

        bearish_count = sum(
            1 for item in multi_results.values()
            if item.get("bias") == "BAISSIER"
        )

        neutral_count = len(multi_results) - bullish_count - bearish_count

        bias_col1, bias_col2, bias_col3 = st.columns(3)

        with bias_col1:
            st.metric("Unités haussières", bullish_count)

        with bias_col2:
            st.metric("Unités baissières", bearish_count)

        with bias_col3:
            st.metric("Unités neutres", neutral_count)

    # -----------------------------------------------------
    # AFFICHAGE DU DERNIER SIGNAL
    # -----------------------------------------------------

    if st.session_state.signal_result is not None:
        signal = st.session_state.signal_result

        st.divider()
        st.subheader("Signal généré")

        if signal["signal"] == "ACHAT":
            st.success("Signal : ACHAT")

        elif signal["signal"] == "VENTE":
            st.error("Signal : VENTE")

        else:
            st.warning("Signal : ATTENDRE")

        sig1, sig2, sig3 = st.columns(3)

        with sig1:
            st.metric(
                "Score de confiance",
                f"{signal.get('confidence', 0)} %"
            )

        with sig2:
            st.metric(
                "Prix d'entrée indicatif",
                format_price(
                    signal.get("entry"),
                    market_name
                )
            )

        with sig3:
            st.metric(
                "Ratio risque/rendement",
                (
                    f"1:{signal['risk_reward']:.1f}"
                    if signal.get("risk_reward") is not None
                    else "—"
                )
            )

        level1, level2, level3 = st.columns(3)

        with level1:
            st.metric(
                "Stop-loss (SL)",
                format_price(signal.get("sl"), market_name)
            )

        with level2:
            st.metric(
                "Take-profit (TP)",
                format_price(signal.get("tp"), market_name)
            )

        with level3:
            st.metric(
                "ID historique",
                (
                    str(st.session_state.signal_id)
                    if st.session_state.signal_id is not None
                    else "—"
                )
            )

        st.markdown("**Motifs de l'analyse**")

        st.write(
            signal.get(
                "reason",
                "Aucun motif supplémentaire."
            )
        )

        st.caption(
            "Signal calculé à : "
            + utc_to_kinshasa(
                signal.get("created_at_utc")
            )
            + " (heure de Kinshasa)"
        )

        st.caption(
            "Bougie utilisée : "
            + utc_to_kinshasa(
                signal.get("candle_time")
            )
        )

        st.warning(
            "Ce signal est une indication technique, pas une "
            "garantie de résultat. Vérifie les spreads, la taille "
            "de position et le risque avant toute entrée."
        )

    else:
        st.info(
            "Clique sur « Lancer l'analyse multi-unités » "
            "pour calculer le premier signal."
        )


# =========================================================
# 20. ONGLET HISTORIQUE
# =========================================================

with tab_history:
    st.subheader("Historique des signaux")

    hist_col1, hist_col2 = st.columns([2, 1])

    with hist_col1:
        history_limit = st.selectbox(
            "Nombre de signaux à afficher",
            options=[20, 50, 100, 250, 500],
            index=2
        )

    with hist_col2:
        if st.button(
            "Actualiser l'historique",
            use_container_width=True
        ):
            st.rerun()

    history = get_signal_history(history_limit)
    if not history.empty:
    if st.button(
        "Enregistrer la consultation des signaux affichés"
    ):
        mark_signals_as_viewed(
            history["id"].tolist()
        )
        st.rerun()

    if history.empty:
        st.info(
            "Aucun signal enregistré. Lance une analyse "
            "pour commencer à remplir l'historique."
        )

    else:
        history_display = history.copy()

        history_display["created_at_utc"] = (
            history_display["created_at_utc"].apply(
                utc_to_kinshasa
            )
        )
        history_display["viewed_at_utc"] = (
    history_display["viewed_at_utc"].apply(
        lambda value: (
            utc_to_kinshasa(value)
            if pd.notna(value)
            else "Pas encore consulté"
        )
    )
        )

        history_display["candle_time_utc"] = (
            history_display["candle_time_utc"].apply(
                utc_to_kinshasa
            )
        )

        history_display = history_display.rename(columns={
            "id": "ID",
            "created_at_utc": "Créé à (Kinshasa)",
            "viewed_at_utc": "Vu à (Kinshasa)",
            "candle_time_utc": "Bougie (Kinshasa)",
            "market": "Marché",
            "timeframe": "Unité",
            "signal": "Signal",
            "confidence": "Confiance (%)",
            "entry": "Entrée",
            "stop_loss": "SL",
            "take_profit": "TP",
            "risk_reward": "Ratio R/R",
            "reason": "Motifs"
        })

        st.dataframe(
            history_display,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Exporter l'historique CSV",
            data=csv_data,
            file_name="forexscope_historique.csv",
            mime="text/csv",
            use_container_width=True
        )


# =========================================================
# 21. ONGLET PARAMÈTRES
# =========================================================

with tab_settings:
    st.subheader("Paramètres et informations")

    st.markdown("**Marchés disponibles**")

    st.write(", ".join(SYMBOLS.keys()))

    st.markdown("**Unités de temps disponibles**")

    st.write(", ".join(INTERVALS.keys()))

    st.markdown("**Indicateurs utilisés**")

    st.write(
        "- EMA 20, EMA 50 et EMA 200\n"
        "- RSI sur 14 périodes\n"
        "- ATR sur 14 périodes\n"
        "- MACD\n"
        "- ADX et indicateurs directionnels\n"
        "- Structure du marché\n"
        "- Balayages simplifiés de liquidité\n"
        "- Confluence multi-unités"
    )

    st.markdown("**Source de données**")

    st.write(
        "Yahoo Finance. La disponibilité, le délai et la "
        "profondeur des données dépendent du marché et de "
        "l'unité de temps."
    )

    st.markdown("**Gestion du risque**")

    st.write(
        "Le SL indicatif est placé à 1,5 ATR de l'entrée "
        "et le TP à 3 ATR, soit un ratio théorique de 1:2. "
        "Ces niveaux ne garantissent pas que l'ordre sera "
        "exécuté au prix demandé."
    )

    st.markdown("**Historique local**")

    st.write(
        "Les signaux sont conservés dans la base SQLite "
        "`forexscope.db` dans l'environnement où tourne "
        "l'application."
    )

    st.info(
        "Cette version n'envoie pas encore automatiquement "
        "de notifications en arrière-plan et ne passe pas "
        "d'ordres chez un courtier."
    )

st.divider()

st.caption(
    "ForexScope — outil d'analyse technique. "
    "Les décisions de trading restent sous la responsabilité "
    "de l'utilisateur."
)

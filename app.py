
# ============================================================
# FOREXSCOPE — PARTIE 1/3
# Configuration, données Deriv, chandeliers et indicateurs
# ============================================================

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import websocket


# ------------------------------------------------------------
# 1. CONFIGURATION GÉNÉRALE
# ------------------------------------------------------------

st.set_page_config(
    page_title="ForexScope",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "ForexScope"
APP_VERSION = "1.0.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forexscope")

# Identifiant public par défaut de l'API Deriv.
# Tu peux le remplacer par ton propre identifiant dans
# les secrets Streamlit ou la variable DERIV_APP_ID.
DERIV_APP_ID = str(
    st.secrets.get(
        "DERIV_APP_ID",
        os.getenv("DERIV_APP_ID", "1089"),
    )
)

DERIV_WS_URL = (
    "wss://ws.derivws.com/websockets/v3"
    f"?app_id={DERIV_APP_ID}"
)

# Unités de temps exprimées en secondes.
TIMEFRAMES = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}

# Symboles Deriv usuels. La disponibilité dépend du compte,
# de la région et des marchés proposés par Deriv.
SYMBOLS = {
    "EUR/USD": "frxEURUSD",
    "GBP/USD": "frxGBPUSD",
    "USD/JPY": "frxUSDJPY",
    "AUD/USD": "frxAUDUSD",
    "USD/CAD": "frxUSDCAD",
    "USD/CHF": "frxUSDCHF",
    "NZD/USD": "frxNZDUSD",
    "EUR/GBP": "frxEURGBP",
    "EUR/JPY": "frxEURJPY",
    "GBP/JPY": "frxGBPJPY",
    "XAU/USD (Or)": "frxXAUUSD",
    "XAG/USD (Argent)": "frxXAGUSD",
}

OHLC_COLUMNS = [
    "time", "open", "high", "low", "close"
]


# ------------------------------------------------------------
# 2. UTILITAIRES
# ------------------------------------------------------------

def utc_now() -> datetime:
    """Renvoie l'heure UTC actuelle."""
    return datetime.now(timezone.utc)


def format_price(value: Any, decimals: int = 5) -> str:
    """Formate un prix sans planter si la valeur est absente."""
    try:
        number = float(value)
        if not np.isfinite(number):
            return "Indisponible"
        return f"{number:.{decimals}f}"
    except (TypeError, ValueError):
        return "Indisponible"


def validate_ohlc(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les chandeliers et contrôle leur cohérence.
    Le temps est conservé en secondes Unix.
    """
    if data is None or data.empty:
        return pd.DataFrame(columns=OHLC_COLUMNS)

    df = data.copy()

    missing = set(OHLC_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes OHLC manquantes : {sorted(missing)}"
        )

    df = df[OHLC_COLUMNS].copy()

    for column in OHLC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=OHLC_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=OHLC_COLUMNS)

    # Les prix doivent être strictement positifs.
    price_columns = ["open", "high", "low", "close"]
    df = df[(df[price_columns] > 0).all(axis=1)]

    # Un chandelier cohérent doit respecter :
    # high >= max(open, close, low)
    # low  <= min(open, close, high)
    df = df[
        (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
    ]

    df = (
        df.drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )

    df["time"] = df["time"].astype("int64")
    return df


# ------------------------------------------------------------
# 3. CONNEXION DERIV WEBSOCKET
# ------------------------------------------------------------

def deriv_request(
    payload: dict[str, Any],
    timeout: int = 15,
) -> dict[str, Any]:
    """
    Effectue une requête ponctuelle à l'API WebSocket Deriv.

    Cette fonction utilise uniquement les données publiques
    demandées. Aucun mot de passe ni jeton de compte n'est requis
    pour récupérer les chandeliers publics.
    """
    ws = None

    try:
        ws = websocket.create_connection(
            DERIV_WS_URL,
            timeout=timeout,
            enable_multithread=False,
        )

        ws.send(json.dumps(payload))

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            raw = ws.recv()

            if not raw:
                continue

            response = json.loads(raw)

            if response.get("error"):
                error = response["error"]
                code = error.get("code", "DerivError")
                message = error.get(
                    "message", "Erreur inconnue de Deriv"
                )
                raise RuntimeError(f"{code}: {message}")

            # Ignorer les messages qui ne correspondent pas
            # à la réponse attendue.
            if (
                "candles" in response
                or "history" in response
                or "active_symbols" in response
                or "ping" in response
            ):
                return response

        raise TimeoutError(
            "Délai dépassé : aucune réponse exploitable de Deriv."
        )

    except Exception as exc:
        logger.warning("Erreur Deriv : %s", exc)
        raise RuntimeError(
            f"Connexion ou réponse Deriv impossible : {exc}"
        ) from exc

    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


# ------------------------------------------------------------
# 4. VÉRIFICATION DES SYMBOLES DISPONIBLES
# ------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_active_symbols() -> pd.DataFrame:
    """
    Récupère la liste des marchés actifs renvoyés par Deriv.
    Le cache limite les requêtes répétées.
    """
    response = deriv_request({
        "active_symbols": "brief",
        "product_type": "basic",
    })

    symbols = response.get("active_symbols", [])

    if not symbols:
        return pd.DataFrame(
            columns=["symbol", "display_name", "market"]
        )

    return pd.DataFrame([
        {
            "symbol": item.get("symbol"),
            "display_name": item.get("display_name"),
            "market": item.get("market"),
        }
        for item in symbols
    ])


# ------------------------------------------------------------
# 5. RÉCUPÉRATION DES CHANDELIERS OHLC
# ------------------------------------------------------------

@st.cache_data(ttl=15, show_spinner=False)
def fetch_candles(
    symbol: str,
    timeframe: str,
    count: int = 500,
) -> pd.DataFrame:
    """
    Récupère les derniers chandeliers d'un instrument.

    Paramètres :
        symbol     : symbole technique Deriv
        timeframe  : unité de temps (M1, M5, M15, H1...)
        count      : nombre de chandeliers demandé

    Retour :
        DataFrame OHLC trié chronologiquement.
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(
            f"Unité de temps non prise en charge : {timeframe}"
        )

    if not symbol or not isinstance(symbol, str):
        raise ValueError("Le symbole de marché est invalide.")

    count = int(count)
    count = max(50, min(count, 1000))

    response = deriv_request({
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "granularity": TIMEFRAMES[timeframe],
        "style": "candles",
    })

    candles = response.get("candles", [])

    if not candles:
        raise RuntimeError(
            f"Aucun chandelier reçu pour {symbol} ({timeframe})."
        )

    rows = []

    for candle in candles:
        rows.append({
            "time": candle.get("epoch"),
            "open": candle.get("open"),
            "high": candle.get("high"),
            "low": candle.get("low"),
            "close": candle.get("close"),
        })

    df = validate_ohlc(pd.DataFrame(rows))

    if len(df) < 50:
        raise RuntimeError(
            f"Données insuffisantes : {len(df)} chandeliers valides."
        )

    return df


# ------------------------------------------------------------
# 6. INDICATEURS TECHNIQUES
# ------------------------------------------------------------

def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    RSI de Wilder, calculé par lissage exponentiel.
    Valeur théorique comprise entre 0 et 100.
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    # Cas particuliers :
    # hausse sans baisse => RSI 100 ;
    # baisse sans hausse => RSI 0 ;
    # prix parfaitement constants => RSI 50.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)

    return rsi


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    ATR : mesure de volatilité en unités de prix,
    et non en pourcentage.
    """
    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Calcule ADX, +DI et -DI.
    L'ADX mesure la force de tendance, pas sa direction.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

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

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    plus_smoothed = plus_dm.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    minus_smoothed = minus_dm.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    plus_di = 100 * plus_smoothed / atr.replace(0, np.nan)
    minus_di = 100 * minus_smoothed / atr.replace(0, np.nan)

    denominator = (plus_di + minus_di).replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / denominator

    adx = dx.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    return pd.DataFrame({
        "ADX": adx,
        "PLUS_DI": plus_di,
        "MINUS_DI": minus_di,
    })


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les indicateurs au DataFrame sans modifier
    les données OHLC d'origine.
    """
    if df is None or df.empty:
        raise ValueError("Impossible d'analyser un tableau vide.")

    result = validate_ohlc(df)

    if len(result) < 50:
        raise ValueError(
            "Il faut au moins 50 chandeliers valides."
        )

    close = result["close"]

    # Moyennes mobiles exponentielles.
    result["EMA20"] = close.ewm(
        span=20, adjust=False
    ).mean()

    result["EMA50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    result["EMA200"] = close.ewm(
        span=200, min_periods=50, adjust=False
    ).mean()

    # RSI.
    result["RSI"] = calculate_rsi(close)

    # MACD standard : 12, 26, 9.
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    result["MACD"] = ema12 - ema26
    result["MACD_SIGNAL"] = result["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    result["MACD_HIST"] = (
        result["MACD"] - result["MACD_SIGNAL"]
    )

    # Volatilité et force de tendance.
    result["ATR"] = calculate_atr(result)

    adx_data = calculate_adx(result)
    result = pd.concat([result, adx_data], axis=1)

    # Prix de clôture de la bougie précédente.
    result["PREVIOUS_CLOSE"] = result["close"].shift(1)

    return result


# FIN DE LA PARTIE 1/3
# La partie 2 ajoutera l'analyse de structure du marché,
# les scores, les verdicts et les niveaux de risque.

# ============================================================
# FOREXSCOPE — PARTIE 2/3
# Structure du marché, liquidité, scoring, signaux et SL/TP
# À coller après la PARTIE 1 dans app.py
# ============================================================

from typing import Optional


# ------------------------------------------------------------
# 1. PARAMÈTRES DU MOTEUR DE DÉCISION
# ------------------------------------------------------------

SIGNAL_CONFIG = {
    "min_candles": 50,
    "min_score": 65.0,
    "min_score_difference": 15.0,
    "min_adx_trend": 20.0,
    "atr_stop_multiplier": 1.5,
    "reward_risk_ratio": 2.0,
    "structure_lookback": 20,
    "sweep_lookback": 10,
}


# ------------------------------------------------------------
# 2. ANALYSE DE STRUCTURE DU MARCHÉ
# ------------------------------------------------------------

def analyze_market_structure(
    df: pd.DataFrame,
    lookback: int = 20,
) -> dict:
    """
    Évalue la structure à partir des plus hauts et plus bas
    récents, sans utiliser de données futures.

    Il s'agit d'une approximation algorithmique de la structure
    du marché, et non d'une identification subjective parfaite
    des swings utilisés en SMC.
    """
    result = {
        "bias": "NEUTRAL",
        "breakout": "NONE",
        "previous_high": None,
        "previous_low": None,
        "higher_highs": False,
        "higher_lows": False,
        "lower_highs": False,
        "lower_lows": False,
    }

    if df is None or len(df) < lookback + 5:
        return result

    # Exclure la bougie la plus récente, potentiellement ouverte.
    closed = df.iloc[:-1].copy()

    if len(closed) < lookback + 5:
        return result

    current = closed.iloc[-1]
    previous_window = closed.iloc[-lookback - 1:-1]

    previous_high = float(previous_window["high"].max())
    previous_low = float(previous_window["low"].min())

    result["previous_high"] = previous_high
    result["previous_low"] = previous_low

    # Comparaison de deux fenêtres antérieures.
    recent = closed.iloc[-10:]
    older = closed.iloc[-20:-10]

    if len(older) == 10 and len(recent) == 10:
        recent_high = float(recent["high"].max())
        older_high = float(older["high"].max())
        recent_low = float(recent["low"].min())
        older_low = float(older["low"].min())

        result["higher_highs"] = recent_high > older_high
        result["higher_lows"] = recent_low > older_low
        result["lower_highs"] = recent_high < older_high
        result["lower_lows"] = recent_low < older_low

    # Cassure confirmée à la clôture d'une bougie.
    if current["close"] > previous_high:
        result["breakout"] = "BULLISH"
    elif current["close"] < previous_low:
        result["breakout"] = "BEARISH"

    bullish_structure = (
        result["higher_highs"] and result["higher_lows"]
    )
    bearish_structure = (
        result["lower_highs"] and result["lower_lows"]
    )

    if bullish_structure:
        result["bias"] = "BULLISH"
    elif bearish_structure:
        result["bias"] = "BEARISH"

    return result


# ------------------------------------------------------------
# 3. DÉTECTION APPROXIMATIVE DES BALAYAGES DE LIQUIDITÉ
# ------------------------------------------------------------

def detect_liquidity_sweep(
    df: pd.DataFrame,
    lookback: int = 10,
) -> dict:
    """
    Détecte un balayage potentiel sur la dernière bougie clôturée.

    Sweep haussier :
      le plus bas dépasse le niveau précédent, puis la clôture
      revient au-dessus de ce niveau.

    Sweep baissier :
      le plus haut dépasse le niveau précédent, puis la clôture
      revient en dessous de ce niveau.

    Cette détection ne prouve pas la présence d'ordres institutionnels.
    """
    result = {
        "type": "NONE",
        "level": None,
        "confirmed": False,
    }

    if df is None or len(df) < lookback + 3:
        return result

    closed = df.iloc[:-1].copy()

    if len(closed) < lookback + 1:
        return result

    candle = closed.iloc[-1]
    previous = closed.iloc[-lookback - 1:-1]

    if previous.empty:
        return result

    prior_low = float(previous["low"].min())
    prior_high = float(previous["high"].max())

    if (
        float(candle["low"]) < prior_low
        and float(candle["close"]) > prior_low
    ):
        result.update({
            "type": "BULLISH_SWEEP",
            "level": prior_low,
            "confirmed": True,
        })

    elif (
        float(candle["high"]) > prior_high
        and float(candle["close"]) < prior_high
    ):
        result.update({
            "type": "BEARISH_SWEEP",
            "level": prior_high,
            "confirmed": True,
        })

    return result


# ------------------------------------------------------------
# 4. ANALYSE D'UNE SEULE UNITÉ DE TEMPS
# ------------------------------------------------------------

def analyze_timeframe(
    raw_df: pd.DataFrame,
    timeframe: str,
) -> dict:
    """
    Produit un diagnostic indépendant pour une unité de temps.

    Le score représente la concordance pondérée des critères.
    Il ne constitue pas une probabilité de gain.
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("Données absentes.")

    if len(raw_df) < SIGNAL_CONFIG["min_candles"]:
        raise ValueError(
            f"{timeframe}: au moins "
            f"{SIGNAL_CONFIG['min_candles']} bougies nécessaires."
        )

    # Ajouter les indicateurs à l'ensemble des données.
    data = calculate_indicators(raw_df)

    # La dernière bougie reçue peut être en cours de formation.
    # On exclut donc cette bougie pour le signal principal.
    closed = data.iloc[:-1].copy()

    if len(closed) < 30:
        raise ValueError(
            f"{timeframe}: pas assez de bougies clôturées."
        )

    last = closed.iloc[-1]
    previous = closed.iloc[-2]

    price = float(last["close"])
    atr = float(last["ATR"]) if pd.notna(last["ATR"]) else np.nan
    rsi = float(last["RSI"]) if pd.notna(last["RSI"]) else np.nan
    adx = float(last["ADX"]) if pd.notna(last["ADX"]) else np.nan

    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    ema200 = (
        float(last["EMA200"])
        if pd.notna(last["EMA200"])
        else np.nan
    )

    macd = (
        float(last["MACD"])
        if pd.notna(last["MACD"])
        else np.nan
    )
    macd_signal = (
        float(last["MACD_SIGNAL"])
        if pd.notna(last["MACD_SIGNAL"])
        else np.nan
    )

    # Unité de prix : celle du symbole analysé.
    # Exemple : ATR = 0.0010 pour EUR/USD.
    if not np.isfinite(atr) or atr <= 0:
        raise ValueError(
            f"{timeframe}: ATR indisponible ou invalide."
        )

    if not np.isfinite(rsi):
        raise ValueError(f"{timeframe}: RSI indisponible.")

    if not np.isfinite(adx):
        raise ValueError(f"{timeframe}: ADX indisponible.")

    structure = analyze_market_structure(closed)
    sweep = detect_liquidity_sweep(closed)

    # Chaque critère contribue à un score haussier ou baissier.
    bullish_score = 0.0
    bearish_score = 0.0
    max_score = 100.0

    evidence = []

    # A. Tendance EMA20 / EMA50 : 20 points.
    if ema20 > ema50:
        bullish_score += 20
        evidence.append("EMA20 au-dessus EMA50")
    elif ema20 < ema50:
        bearish_score += 20
        evidence.append("EMA20 sous EMA50")

    # B. Position du prix par rapport à EMA200 : 15 points.
    if np.isfinite(ema200):
        if price > ema200:
            bullish_score += 15
            evidence.append("Prix au-dessus EMA200")
        elif price < ema200:
            bearish_score += 15
            evidence.append("Prix sous EMA200")

    # C. RSI : 15 points.
    # On évite de traiter automatiquement une zone de surachat
    # comme une vente ou une zone de survente comme un achat.
    if 52 <= rsi <= 68:
        bullish_score += 15
        evidence.append("RSI favorable aux acheteurs")
    elif 32 <= rsi <= 48:
        bearish_score += 15
        evidence.append("RSI favorable aux vendeurs")

    # D. MACD : 15 points.
    if np.isfinite(macd) and np.isfinite(macd_signal):
        if macd > macd_signal:
            bullish_score += 15
            evidence.append("MACD haussier")
        elif macd < macd_signal:
            bearish_score += 15
            evidence.append("MACD baissier")

    # E. Direction DI : 10 points.
    plus_di = float(last["PLUS_DI"])
    minus_di = float(last["MINUS_DI"])

    if plus_di > minus_di:
        bullish_score += 10
        evidence.append("+DI supérieur à -DI")
    elif minus_di > plus_di:
        bearish_score += 10
        evidence.append("-DI supérieur à +DI")

    # F. Structure du marché : 15 points.
    if structure["bias"] == "BULLISH":
        bullish_score += 15
        evidence.append("Structure haussière")
    elif structure["bias"] == "BEARISH":
        bearish_score += 15
        evidence.append("Structure baissière")

    # G. Balayage de liquidité : 10 points.
    if sweep["type"] == "BULLISH_SWEEP":
        bullish_score += 10
        evidence.append("Balayage potentiel sous un niveau")
    elif sweep["type"] == "BEARISH_SWEEP":
        bearish_score += 10
        evidence.append("Balayage potentiel au-dessus d'un niveau")

    # Calcul du score de concordance directionnelle.
    bullish_pct = bullish_score / max_score * 100
    bearish_pct = bearish_score / max_score * 100
    difference = abs(bullish_score - bearish_score)

    # Les signaux exigent un score suffisant et une différence
    # minimale entre les deux directions.
    minimum_score = SIGNAL_CONFIG["min_score"]
    minimum_difference = SIGNAL_CONFIG["min_score_difference"]

    signal = "ATTENTE"
    direction = "NEUTRAL"

    if (
        bullish_pct >= minimum_score
        and difference >= minimum_difference
    ):
        signal = "ACHAT"
        direction = "BULLISH"

    elif (
        bearish_pct >= minimum_score
        and difference >= minimum_difference
    ):
        signal = "VENTE"
        direction = "BEARISH"

    # Un ADX faible indique une tendance peu marquée.
    # Ce critère ne modifie pas les scores, mais empêche
    # un signal directionnel en l'absence de force suffisante.
    if adx < SIGNAL_CONFIG["min_adx_trend"]:
        signal = "ATTENTE"
        direction = "NEUTRAL"
        evidence.append("ADX faible : tendance peu affirmée")

    # Un marché trop étroit ou un ATR nul invalide le risque.
    if atr <= 0 or not np.isfinite(atr):
        signal = "ATTENTE"
        direction = "NEUTRAL"

    # Niveaux de risque : seulement si le signal est validé.
    stop_loss = None
    take_profit = None
    risk_distance = None
    reward_distance = None
    risk_reward = None

    if signal in ("ACHAT", "VENTE"):
        multiplier = SIGNAL_CONFIG["atr_stop_multiplier"]
        rr_target = SIGNAL_CONFIG["reward_risk_ratio"]

        # Distance minimale entre entrée et SL, exprimée
        # dans l'unité de prix de l'instrument.
        risk_distance = atr * multiplier

        if signal == "ACHAT":
            stop_loss = price - risk_distance
            take_profit = price + risk_distance * rr_target
        else:
            stop_loss = price + risk_distance
            take_profit = price - risk_distance * rr_target

        reward_distance = abs(take_profit - price)
        risk_reward = reward_distance / risk_distance

    # Pour l'affichage, score de la direction dominante.
    directional_score = max(bullish_pct, bearish_pct)

    return {
        "timeframe": timeframe,
        "signal": signal,
        "direction": direction,
        "price": price,
        "rsi": rsi,
        "adx": adx,
        "atr": atr,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "macd": macd,
        "macd_signal": macd_signal,
        "bullish_score": round(bullish_pct, 1),
        "bearish_score": round(bearish_pct, 1),
        "directional_score": round(directional_score, 1),
        "score_difference": round(difference, 1),
        "structure": structure,
        "sweep": sweep,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_distance": risk_distance,
        "reward_distance": reward_distance,
        "risk_reward": risk_reward,
        "evidence": evidence,
        "candle_time": int(last["time"]),
    }


# ------------------------------------------------------------
# 5. ANALYSE MULTI-UNITÉS DE TEMPS
# ------------------------------------------------------------

def analyze_multiple_timeframes(
    candles_by_timeframe: dict[str, pd.DataFrame],
    primary_timeframe: str = "M15",
) -> dict:
    """
    Analyse les unités de temps séparément puis calcule
    un consensus. Une unité manquante n'est pas remplacée
    par des données simulées.

    La direction du consensus dépend des unités réellement
    analysées avec succès.
    """
    results = {}
    errors = {}

    for timeframe, candles in candles_by_timeframe.items():
        try:
            results[timeframe] = analyze_timeframe(
                candles, timeframe
            )
        except Exception as exc:
            errors[timeframe] = str(exc)

    if not results:
        return {
            "primary_timeframe": primary_timeframe,
            "primary": None,
            "consensus": "ATTENTE",
            "consensus_score": 0.0,
            "agreement_pct": 0.0,
            "results": {},
            "errors": errors,
            "reason": "Aucune unité de temps exploitable.",
        }

    primary = results.get(primary_timeframe)

    # Pondération explicite par horizon :
    # les unités longues définissent davantage le contexte,
    # mais ne garantissent pas la direction future.
    timeframe_weights = {
        "M1": 0.5,
        "M5": 0.75,
        "M15": 1.0,
        "M30": 1.1,
        "H1": 1.25,
        "H4": 1.5,
        "D1": 1.75,
    }

    bullish_weight = 0.0
    bearish_weight = 0.0
    total_weight = 0.0

    for timeframe, result in results.items():
        weight = timeframe_weights.get(timeframe, 1.0)
        total_weight += weight

        if result["signal"] == "ACHAT":
            bullish_weight += weight
        elif result["signal"] == "VENTE":
            bearish_weight += weight

    if total_weight <= 0:
        agreement_pct = 0.0
        consensus_score = 0.0
        consensus = "ATTENTE"
    else:
        dominant_weight = max(bullish_weight, bearish_weight)

        agreement_pct = dominant_weight / total_weight * 100

        net_weight = abs(bullish_weight - bearish_weight)
        consensus_score = net_weight / total_weight * 100

        if (
            bullish_weight > bearish_weight
            and agreement_pct >= 60
        ):
            consensus = "ACHAT"
        elif (
            bearish_weight > bullish_weight
            and agreement_pct >= 60
        ):
            consensus = "VENTE"
        else:
            consensus = "ATTENTE"

    # Le consensus ne doit pas masquer un désaccord avec
    # l'unité principale choisie par l'utilisateur.
    if primary is not None:
        if (
            primary["signal"] in ("ACHAT", "VENTE")
            and consensus in ("ACHAT", "VENTE")
            and primary["signal"] != consensus
        ):
            consensus = "ATTENTE"
            reason = (
                "Désaccord entre l'unité principale et le consensus."
            )
        else:
            reason = (
                "Consensus calculé à partir des unités disponibles."
            )
    else:
        reason = (
            "Unité principale indisponible ; "
            "consensus calculé sur les autres unités."
        )

    return {
        "primary_timeframe": primary_timeframe,
        "primary": primary,
        "consensus": consensus,
        "consensus_score": round(consensus_score, 1),
        "agreement_pct": round(agreement_pct, 1),
        "results": results,
        "errors": errors,
        "reason": reason,
    }


# ------------------------------------------------------------
# 6. PRÉSENTATION DES RÉSULTATS POUR LE TABLEAU DE BORD
# ------------------------------------------------------------

def signal_summary(analysis: dict) -> pd.DataFrame:
    """Transforme les résultats multi-unités en tableau."""
    rows = []

    for timeframe, result in analysis.get("results", {}).items():
        rows.append({
            "Unité": timeframe,
            "Signal": result["signal"],
            "Prix": result["price"],
            "RSI": result["rsi"],
            "ADX": result["adx"],
            "Score achat (%)": result["bullish_score"],
            "Score vente (%)": result["bearish_score"],
            "SL": result["stop_loss"],
            "TP": result["take_profit"],
            "Risque/rendement": result["risk_reward"],
        })

    return pd.DataFrame(rows)


# FIN DE LA PARTIE 2/3
# La partie 3 intégrera ces fonctions dans Streamlit :
# interface, graphiques, sélection d'actifs, actualisation,
# historique persistant et gestion des erreurs.

# Imports supplémentaires pour la partie 3
import sqlite3
from pathlib import Path

import plotly.graph_objects as go

# ============================================================
# FOREXSCOPE — PARTIE 3/3
# Interface, graphiques, historique et actualisation
# ============================================================

# ------------------------------------------------------------
# 1. STOCKAGE PERSISTANT DES SIGNAUX
# ------------------------------------------------------------

DB_PATH = Path(
    os.getenv("FOREXSCOPE_DB_PATH", "forexscope.db")
)


def init_database() -> None:
    """Crée la base locale si elle n'existe pas."""
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                candle_time INTEGER NOT NULL,
                signal TEXT NOT NULL,
                entry REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                score REAL NOT NULL,
                rsi REAL,
                adx REAL,
                UNIQUE(symbol, timeframe, candle_time)
            )
        """)
        conn.commit()


def save_signal(
    symbol: str,
    result: dict,
) -> None:
    """
    Enregistre les signaux ACHAT et VENTE une seule fois
    par symbole, unité de temps et bougie clôturée.

    ATTENTE n'est pas enregistré comme signal de trading.
    """
    if result.get("signal") not in ("ACHAT", "VENTE"):
        return

    values = (
        utc_now().isoformat(),
        symbol,
        result["timeframe"],
        int(result["candle_time"]),
        result["signal"],
        float(result["price"]),
        float(result["stop_loss"]),
        float(result["take_profit"]),
        float(result["directional_score"]),
        float(result["rsi"]),
        float(result["adx"]),
    )

    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute("""
            INSERT OR IGNORE INTO signals (
                created_at, symbol, timeframe, candle_time,
                signal, entry, stop_loss, take_profit,
                score, rsi, adx
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, values)
        conn.commit()


def load_signal_history(
    symbol: Optional[str] = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Charge les derniers signaux sauvegardés."""
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        if symbol:
            df = pd.read_sql_query(
                """
                SELECT *
                FROM signals
                WHERE symbol = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                conn,
                params=(symbol, int(limit)),
            )
        else:
            df = pd.read_sql_query(
                """
                SELECT *
                FROM signals
                ORDER BY id DESC
                LIMIT ?
                """,
                conn,
                params=(int(limit),),
            )

    return df


init_database()


# ------------------------------------------------------------
# 2. OUTILS D'AFFICHAGE
# ------------------------------------------------------------

def display_price(
    value: Any,
    symbol: str,
) -> str:
    """Choisit une précision d'affichage indicative."""
    if value is None:
        return "—"

    try:
        number = float(value)

        if not np.isfinite(number):
            return "—"

        # Affichage indicatif uniquement.
        if "JPY" in symbol:
            decimals = 3
        elif "XAU" in symbol or "XAG" in symbol:
            decimals = 2
        else:
            decimals = 5

        return f"{number:.{decimals}f}"

    except (TypeError, ValueError):
        return "—"


def build_candlestick_chart(
    raw_df: pd.DataFrame,
    symbol_label: str,
    timeframe: str,
    result: Optional[dict] = None,
) -> go.Figure:
    """Construit un graphique OHLC avec les niveaux du signal."""
    df = calculate_indicators(raw_df).copy()

    df["datetime"] = pd.to_datetime(
        df["time"], unit="s", utc=True
    )

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["datetime"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Prix",
    ))

    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=df["EMA20"],
        name="EMA 20",
        mode="lines",
    ))

    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=df["EMA50"],
        name="EMA 50",
        mode="lines",
    ))

    # N'afficher EMA200 que lorsque suffisamment de données
    # ont été calculées.
    if df["EMA200"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=df["EMA200"],
            name="EMA 200",
            mode="lines",
        ))

    if result and result.get("signal") in ("ACHAT", "VENTE"):
        entry = result.get("price")
        sl = result.get("stop_loss")
        tp = result.get("take_profit")

        if entry is not None:
            fig.add_hline(
                y=entry,
                line_dash="solid",
                annotation_text="Entrée indicative",
            )

        if sl is not None:
            fig.add_hline(
                y=sl,
                line_dash="dash",
                annotation_text="Stop-loss",
            )

        if tp is not None:
            fig.add_hline(
                y=tp,
                line_dash="dash",
                annotation_text="Take-profit",
            )

    fig.update_layout(
        title=f"{symbol_label} — {timeframe}",
        xaxis_title="Temps (UTC)",
        yaxis_title="Prix",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


# ------------------------------------------------------------
# 3. INITIALISATION DE L'ÉTAT STREAMLIT
# ------------------------------------------------------------

if "last_sync" not in st.session_state:
    st.session_state["last_sync"] = None

if "last_results" not in st.session_state:
    st.session_state["last_results"] = {}

if "last_errors" not in st.session_state:
    st.session_state["last_errors"] = {}


# ------------------------------------------------------------
# 4. EN-TÊTE
# ------------------------------------------------------------

st.title("ForexScope")
st.caption(
    "Analyse technique multi-unités de temps — "
    "données publiques Deriv"
)

header_left, header_right = st.columns([3, 1])

with header_left:
    st.write(
        "Analyse des marchés, concordance des signaux "
        "et gestion indicative du risque."
    )

with header_right:
    st.caption(f"Version {APP_VERSION}")

st.divider()


# ------------------------------------------------------------
# 5. BARRE LATÉRALE : PARAMÈTRES
# ------------------------------------------------------------

st.sidebar.header("Configuration")

symbol_label = st.sidebar.selectbox(
    "Instrument",
    options=list(SYMBOLS.keys()),
    index=0,
)

symbol = SYMBOLS[symbol_label]

primary_timeframe = st.sidebar.selectbox(
    "Unité de temps principale",
    options=list(TIMEFRAMES.keys()),
    index=list(TIMEFRAMES.keys()).index("M15"),
)

selected_timeframes = st.sidebar.multiselect(
    "Unités pour l'analyse multi-TF",
    options=list(TIMEFRAMES.keys()),
    default=["M5", "M15", "H1", "H4", "D1"],
)

if primary_timeframe not in selected_timeframes:
    selected_timeframes = [
        primary_timeframe,
        *selected_timeframes,
    ]

selected_timeframes = list(dict.fromkeys(selected_timeframes))

candle_count = st.sidebar.select_slider(
    "Nombre de chandeliers",
    options=[100, 200, 300, 500, 750, 1000],
    value=300,
)

st.sidebar.caption(
    "Une actualisation récupère les données des unités "
    "sélectionnées. Les limites de l'API peuvent varier."
)

refresh = st.sidebar.button(
    "Actualiser les données",
    type="primary",
    use_container_width=True,
)

if refresh:
    # Invalider le cache de données avant la récupération.
    fetch_candles.clear()
    get_active_symbols.clear()


# ------------------------------------------------------------
# 6. VÉRIFICATION DES MARCHÉS
# ------------------------------------------------------------

with st.expander("Vérifier la disponibilité des marchés Deriv"):
    if st.button("Charger les symboles disponibles"):
        try:
            active_df = get_active_symbols()

            if active_df.empty:
                st.warning(
                    "Deriv n'a renvoyé aucun symbole exploitable."
                )
            else:
                st.dataframe(
                    active_df,
                    use_container_width=True,
                    hide_index=True,
                )

                available = set(active_df["symbol"].dropna())

                if symbol in available:
                    st.success(
                        f"{symbol_label} est présent dans "
                        "la liste des symboles actifs."
                    )
                else:
                    st.warning(
                        f"{symbol} n'apparaît pas dans la liste "
                        "reçue. Vérifie le symbole exact."
                    )

        except Exception as exc:
            st.error(f"Vérification impossible : {exc}")


# ------------------------------------------------------------
# 7. RÉCUPÉRATION DES DONNÉES ET ANALYSE
# ------------------------------------------------------------

candles_by_timeframe = {}
fetch_errors = {}

with st.spinner("Récupération et analyse des marchés..."):
    for timeframe in selected_timeframes:
        try:
            candles_by_timeframe[timeframe] = fetch_candles(
                symbol=symbol,
                timeframe=timeframe,
                count=candle_count,
            )
        except Exception as exc:
            fetch_errors[timeframe] = str(exc)
            logger.warning(
                "Échec de récupération %s/%s : %s",
                symbol,
                timeframe,
                exc,
            )

analysis = analyze_multiple_timeframes(
    candles_by_timeframe,
    primary_timeframe=primary_timeframe,
)

st.session_state["last_results"] = analysis
st.session_state["last_errors"] = fetch_errors
st.session_state["last_sync"] = utc_now().isoformat()

for timeframe, result in analysis.get("results", {}).items():
    try:
        save_signal(symbol, result)
    except Exception as exc:
        logger.exception("Impossible de sauvegarder le signal.")
        st.warning(
            f"Le signal {timeframe} n'a pas pu être enregistré : "
            f"{exc}"
        )


# ------------------------------------------------------------
# 8. ÉTAT DE LA CONNEXION ET SYNCHRONISATION
# ------------------------------------------------------------

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if candles_by_timeframe:
        st.success(
            f"Données reçues : {len(candles_by_timeframe)} "
            f"/ {len(selected_timeframes)} unités"
        )
    else:
        st.error("Aucune unité de temps récupérée")

with status_col2:
    st.metric(
        "Unités analysées",
        len(analysis.get("results", {})),
    )

with status_col3:
    sync_time = st.session_state.get("last_sync")

    if sync_time:
        st.caption("Dernière synchronisation (UTC)")
        st.write(sync_time.replace("T", " ")[:19])

if fetch_errors:
    with st.expander("Détails des erreurs de données"):
        for timeframe, message in fetch_errors.items():
            st.error(f"{timeframe} : {message}")


# ------------------------------------------------------------
# 9. INDICATEURS PRINCIPAUX
# ------------------------------------------------------------

primary = analysis.get("primary")
consensus = analysis.get("consensus", "ATTENTE")

if primary is None:
    st.warning(
        "L'unité principale n'a pas pu être analysée. "
        "Vérifie les données et les erreurs de connexion."
    )
else:
    score = primary["directional_score"]

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Prix de clôture analysé",
            display_price(primary["price"], symbol),
        )

    with metric2:
        st.metric(
            f"Signal {primary_timeframe}",
            primary["signal"],
        )

    with metric3:
        st.metric(
            "RSI (14)",
            f"{primary['rsi']:.1f}",
        )

    with metric4:
        st.metric(
            "ADX (14)",
            f"{primary['adx']:.1f}",
        )

    st.subheader("Décision multi-unités de temps")

    decision_col1, decision_col2 = st.columns(2)

    with decision_col1:
        st.metric("Consensus", consensus)
        st.caption(analysis.get("reason", ""))

    with decision_col2:
        st.metric(
            "Concordance directionnelle",
            f"{analysis.get('agreement_pct', 0):.1f} %",
        )
        st.caption(
            "Part pondérée des unités ayant le même signal "
            "directionnel dominant ; ce n'est pas une probabilité "
            "de réussite."
        )

    st.caption(
        f"Score directionnel de l'unité principale : "
        f"{score:.1f} / 100"
    )

    if primary["signal"] in ("ACHAT", "VENTE"):
        st.subheader("Niveaux de risque indicatifs")

        risk1, risk2, risk3, risk4 = st.columns(4)

        with risk1:
            st.metric(
                "Entrée théorique",
                display_price(primary["price"], symbol),
            )

        with risk2:
            st.metric(
                "Stop-loss",
                display_price(primary["stop_loss"], symbol),
            )

        with risk3:
            st.metric(
                "Take-profit",
                display_price(primary["take_profit"], symbol),
            )

        with risk4:
            rr = primary.get("risk_reward")
            st.metric(
                "Ratio rendement/risque",
                f"{rr:.2f}" if rr is not None else "—",
            )

        st.warning(
            "Ces niveaux sont calculés sur le prix de clôture "
            "analysé. Ils ne tiennent pas automatiquement compte "
            "du spread, du glissement, des commissions ou des "
            "distances minimales imposées par le courtier."
        )
    else:
        st.info(
            "Aucun signal directionnel suffisamment confirmé "
            "pour l'unité principale. Le verdict est ATTENTE."
        )


# ------------------------------------------------------------
# 10. GRAPHIQUE PRINCIPAL
# ------------------------------------------------------------

st.divider()
st.subheader("Graphique du marché")

primary_candles = candles_by_timeframe.get(primary_timeframe)

if primary_candles is not None and not primary_candles.empty:
    try:
        fig = build_candlestick_chart(
            primary_candles,
            symbol_label,
            primary_timeframe,
            primary,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"Impossible de construire le graphique : {exc}")
else:
    st.warning(
        "Le graphique n'est pas disponible car les données "
        "de l'unité principale n'ont pas été reçues."
    )


# ------------------------------------------------------------
# 11. TABLEAU MULTI-UNITÉS
# ------------------------------------------------------------

st.divider()
st.subheader("Comparaison multi-unités")

summary_df = signal_summary(analysis)

if not summary_df.empty:
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Prix": st.column_config.NumberColumn(
                format="%.5f"
            ),
            "RSI": st.column_config.NumberColumn(
                format="%.1f"
            ),
            "ADX": st.column_config.NumberColumn(
                format="%.1f"
            ),
            "Score achat (%)": st.column_config.NumberColumn(
                format="%.1f"
            ),
            "Score vente (%)": st.column_config.NumberColumn(
                format="%.1f"
            ),
            "SL": st.column_config.NumberColumn(
                format="%.5f"
            ),
            "TP": st.column_config.NumberColumn(
                format="%.5f"
            ),
            "Risque/rendement": st.column_config.NumberColumn(
                format="%.2f"
            ),
        },
    )
else:
    st.info("Aucun résultat multi-unités disponible.")


# ------------------------------------------------------------
# 12. JUSTIFICATION DU SIGNAL
# ------------------------------------------------------------

if primary:
    st.subheader("Critères observés")

    if primary["evidence"]:
        for item in primary["evidence"]:
            st.write(f"- {item}")
    else:
        st.write(
            "Aucun critère directionnel suffisamment marqué."
        )

    structure = primary["structure"]
    sweep = primary["sweep"]

    detail1, detail2 = st.columns(2)

    with detail1:
        st.markdown("**Structure du marché**")
        st.write(f"Orientation : {structure['bias']}")
        st.write(f"Breakout : {structure['breakout']}")

        if structure["previous_high"] is not None:
            st.write(
                "Résistance de référence : "
                + display_price(
                    structure["previous_high"], symbol
                )
            )

        if structure["previous_low"] is not None:
            st.write(
                "Support de référence : "
                + display_price(
                    structure["previous_low"], symbol
                )
            )

    with detail2:
        st.markdown("**Liquidité**")
        st.write(f"Détection : {sweep['type']}")

        if sweep["level"] is not None:
            st.write(
                "Niveau détecté : "
                + display_price(sweep["level"], symbol)
            )

        st.caption(
            "Les balayages sont des détections techniques "
            "approximatives et ne prouvent pas l'activité "
            "d'acteurs institutionnels."
        )


# ------------------------------------------------------------
# 13. HISTORIQUE PERSISTANT
# ------------------------------------------------------------

st.divider()
st.subheader("Historique des signaux sauvegardés")

history_filter = st.selectbox(
    "Filtrer l'historique",
    options=["Tous les instruments", symbol_label],
)

history_symbol = None if history_filter == "Tous les instruments" else symbol

try:
    history = load_signal_history(
        symbol=history_symbol,
        limit=200,
    )

    if history.empty:
        st.info(
            "Aucun signal ACHAT ou VENTE n'a encore été enregistré. "
            "Les signaux ATTENTE ne sont pas sauvegardés."
        )
    else:
        history["candle_time_utc"] = pd.to_datetime(
            history["candle_time"],
            unit="s",
            utc=True,
        )

        st.dataframe(
            history[
                [
                    "created_at",
                    "symbol",
                    "timeframe",
                    "signal",
                    "entry",
                    "stop_loss",
                    "take_profit",
                    "score",
                    "candle_time_utc",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Exporter l'historique CSV",
        data=csv_bytes,
            file_name="forexscope_signals.csv",
            mime="text/csv",
        )

except Exception as exc:
    st.error(f"Erreur lors du chargement de l'historique : {exc}")


# ------------------------------------------------------------
# 14. PIED DE PAGE
# ------------------------------------------------------------

st.divider()

st.caption(
    "ForexScope est un outil d'analyse technique. "
    "Il ne garantit ni bénéfice ni taux de réussite. "
    "Les signaux doivent être validés en compte démo."
)

st.caption(
    "Données : Deriv WebSocket. "
    "Les disponibilités, prix et conditions de négociation "
    "dépendent de l'instrument et du compte."
)

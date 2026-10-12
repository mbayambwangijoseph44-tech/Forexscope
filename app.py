
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ForexScope",
    page_icon="📊",
    layout="wide",
)

APP_NAME = "ForexScope"
DB_PATH = "forexscope.db"

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
    "Bitcoin / USD": "BTC-USD",
}

TIMEFRAMES = {
    "M1": {"interval": "1m", "period": "7d"},
    "M5": {"interval": "5m", "period": "60d"},
    "M15": {"interval": "15m", "period": "60d"},
    "M30": {"interval": "30m", "period": "60d"},
    "H1": {"interval": "60m", "period": "2y"},
    "H4": {"interval": "60m", "period": "2y"},
    "D1": {"interval": "1d", "period": "5y"},
}

TIMEFRAME_ORDER = ["M5", "M15", "M30", "H1", "H4", "D1"]


# ============================================================
# NORMALISATION DES BOUGIES
# ============================================================

def validate_ohlc(data):
    """Normalise les colonnes et vérifie les bougies OHLC."""
    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower() for c in df.columns]

    required = ["open", "high", "low", "close"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    if "time" not in df.columns:
        df["time"] = df.index

    df["time"] = pd.to_datetime(
        df["time"], utc=True, errors="coerce"
    )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0

    df["volume"] = pd.to_numeric(
        df["volume"], errors="coerce"
    ).fillna(0)

    df = df.dropna(
        subset=["time", "open", "high", "low", "close"]
    )

    df = df[
        (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
        & (df["close"] > 0)
    ]

    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last")
    return df.reset_index(drop=True)


# ============================================================
# SOURCE YAHOO FINANCE
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles(symbol, timeframe, count=300):
    """
    Télécharge les bougies Yahoo Finance.
    Pour H4, les bougies horaires sont regroupées en bougies de 4 heures.
    """
    if timeframe not in TIMEFRAMES:
        return pd.DataFrame()

    settings = TIMEFRAMES[timeframe]

    try:
        raw = yf.download(
            tickers=symbol,
            period=settings["period"],
            interval=settings["interval"],
            auto_adjust=False,
            progress=False,
            threads=False,
            multi_level_index=False,
        )

        if raw is None or raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.columns = [str(c).lower() for c in raw.columns]

        needed = ["open", "high", "low", "close"]
        if not all(c in raw.columns for c in needed):
            return pd.DataFrame()

        raw = raw.copy()
        raw.index = pd.to_datetime(raw.index, utc=True)

        if timeframe == "H4":
            raw = raw.resample(
                "4h", label="left", closed="left"
            ).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                **(
                    {"volume": "sum"}
                    if "volume" in raw.columns
                    else {}
                ),
            }).dropna(subset=needed)

        raw = raw.tail(int(count)).copy()
        raw["time"] = raw.index

        if "volume" not in raw.columns:
            raw["volume"] = 0

        return validate_ohlc(raw.reset_index(drop=True))

    except Exception:
        return pd.DataFrame()


def get_connection_status(symbol):
    """Vérifie si Yahoo Finance fournit des données pour le symbole."""
    try:
        check = yf.download(
            tickers=symbol,
            period="5d",
            interval="1d",
            progress=False,
            threads=False,
            auto_adjust=False,
            multi_level_index=False,
        )

        if check is not None and not check.empty:
            return {
                "ok": True,
                "message": "Connexion Yahoo Finance opérationnelle.",
            }

        return {
            "ok": False,
            "message": (
                "Yahoo Finance n'a retourné aucune donnée pour ce symbole."
            ),
        }

    except Exception as exc:
        return {
            "ok": False,
            "message": f"Erreur Yahoo Finance : {exc}",
        }


# ============================================================
# INDICATEURS TECHNIQUES
# ============================================================

def add_indicators(data):
    df = validate_ohlc(data)

    if df.empty:
        return df

    df["ema20"] = df["close"].ewm(
        span=20, adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    df["ema200"] = df["close"].ewm(
        span=200, adjust=False
    ).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    ema12 = df["close"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26, adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    return df

# ============================================================
# STRUCTURE DU MARCHE ET LIQUIDITE
# ============================================================

def analyze_market_structure(data, lookback=20):
    if data is None or len(data) < lookback + 5:
        return {
            "trend": "Indéterminée",
            "structure": "Données insuffisantes",
            "liquidity": "Non déterminée",
            "sweep": None,
        }

    df = data.copy()
    recent = df.tail(lookback)
    previous = df.iloc[-lookback - 1:-1]

    last = df.iloc[-1]

    recent_high = recent["high"].max()
    recent_low = recent["low"].min()
    previous_high = previous["high"].max()
    previous_low = previous["low"].min()

    if last["close"] > previous_high:
        structure = "Cassure haussière (BOS)"
    elif last["close"] < previous_low:
        structure = "Cassure baissière (BOS)"
    else:
        structure = "Structure en consolidation"

    if last["ema20"] > last["ema50"]:
        trend = "Haussière"
    elif last["ema20"] < last["ema50"]:
        trend = "Baissière"
    else:
        trend = "Neutre"

    sweep = None

    # Détection simplifiée d'une prise de liquidité :
    # mèche sous le plus bas précédent puis clôture au-dessus.
    if (
        last["low"] < previous_low
        and last["close"] > previous_low
    ):
        sweep = "Bullish liquidity sweep"

    elif (
        last["high"] > previous_high
        and last["close"] < previous_high
    ):
        sweep = "Bearish liquidity sweep"

    liquidity = (
        f"Zone haute : {recent_high:.5f} | "
        f"Zone basse : {recent_low:.5f}"
    )

    return {
        "trend": trend,
        "structure": structure,
        "liquidity": liquidity,
        "sweep": sweep,
        "recent_high": float(recent_high),
        "recent_low": float(recent_low),
    }


def analyze_timeframe(symbol, timeframe, count=300):
    candles = fetch_candles(symbol, timeframe, count)

    if candles.empty or len(candles) < 60:
        return {
            "ok": False,
            "timeframe": timeframe,
            "message": "Historique insuffisant pour cette unité.",
        }

    df = add_indicators(candles)
    df = df.dropna(subset=["ema20", "ema50", "rsi", "atr"])

    if len(df) < 5:
        return {
            "ok": False,
            "timeframe": timeframe,
            "message": "Indicateurs non disponibles.",
        }

    last = df.iloc[-1]
    structure = analyze_market_structure(df)

    if last["ema20"] > last["ema50"]:
        direction = "Achat"
    elif last["ema20"] < last["ema50"]:
        direction = "Vente"
    else:
        direction = "Neutre"

    return {
        "ok": True,
        "timeframe": timeframe,
        "direction": direction,
        "trend": structure["trend"],
        "structure": structure["structure"],
        "liquidity": structure["liquidity"],
        "sweep": structure["sweep"],
        "rsi": float(last["rsi"]),
        "atr": float(last["atr"]),
        "close": float(last["close"]),
        "ema20": float(last["ema20"]),
        "ema50": float(last["ema50"]),
        "macd": float(last["macd"]),
        "macd_signal": float(last["macd_signal"]),
        "data": df,
    }


# ============================================================
# ANALYSE MULTI-UNITES DE TEMPS
# ============================================================

def multi_timeframe_analysis(symbol):
    results = {}

    for timeframe in TIMEFRAME_ORDER:
        results[timeframe] = analyze_timeframe(
            symbol, timeframe
        )

    valid = [
        result for result in results.values()
        if result.get("ok")
    ]

    if not valid:
        return {
            "ok": False,
            "message": "Aucune unité de temps exploitable.",
            "timeframes": results,
        }

    buys = sum(
        result["direction"] == "Achat"
        for result in valid
    )

    sells = sum(
        result["direction"] == "Vente"
        for result in valid
    )

    total = len(valid)

    if buys > sells:
        consensus = "Achat"
        agreement = buys / total * 100
    elif sells > buys:
        consensus = "Vente"
        agreement = sells / total * 100
    else:
        consensus = "Attendre"
        agreement = 50.0

    return {
        "ok": True,
        "timeframes": results,
        "consensus": consensus,
        "agreement": round(agreement, 1),
        "valid_timeframes": total,
    }


# ============================================================
# GENERATION DU SIGNAL ET NIVEAUX DE RISQUE
# ============================================================

def generate_trading_signal(symbol, timeframe):
    analysis = analyze_timeframe(symbol, timeframe)

    if not analysis.get("ok"):
        return {
            "ok": False,
            "signal": "Attendre",
            "message": analysis.get(
                "message", "Analyse indisponible."
            ),
        }

    df = analysis["data"]
    last = df.iloc[-1]

    direction = analysis["direction"]
    rsi = analysis["rsi"]
    atr = analysis["atr"]
    price = analysis["close"]

    if not np.isfinite(atr) or atr <= 0:
        return {
            "ok": False,
            "signal": "Attendre",
            "message": "ATR indisponible : niveaux de risque non calculables.",
        }

    # Vérification de la dynamique et des conditions RSI.
    if direction == "Achat":
        valid_rsi = 45 <= rsi <= 70
        valid_macd = last["macd"] >= last["macd_signal"]

    elif direction == "Vente":
        valid_rsi = 30 <= rsi <= 55
        valid_macd = last["macd"] <= last["macd_signal"]

    else:
        valid_rsi = False
        valid_macd = False

    structure = analyze_market_structure(df)
    sweep = structure.get("sweep")

    # Score indicatif, pas une probabilité statistique.
    score = 0
    score += 30 if direction != "Neutre" else 0
    score += 20 if valid_rsi else 0
    score += 20 if valid_macd else 0
    score += 20 if (
        (direction == "Achat" and sweep == "Bullish liquidity sweep")
        or (
            direction == "Vente"
            and sweep == "Bearish liquidity sweep"
        )
    ) else 0
    score += 10 if structure["structure"].startswith("Cassure") else 0

    if (
        direction == "Neutre"
        or not valid_rsi
        or not valid_macd
        or score < 50
    ):
        signal = "Attendre"
    else:
        signal = direction

    entry = price

    if signal == "Achat":
        stop_loss = entry - 1.5 * atr
        take_profit = entry + 3.0 * atr
    elif signal == "Vente":
        stop_loss = entry + 1.5 * atr
        take_profit = entry - 3.0 * atr
    else:
        stop_loss = None
        take_profit = None

    return {
        "ok": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "rsi": rsi,
        "atr": atr,
        "score": score,
        "trend": analysis["trend"],
        "structure": analysis["structure"],
        "liquidity": analysis["liquidity"],
        "sweep": sweep,
        "close_time": str(last["time"]),
        "message": (
            "Signal indicatif basé sur les données et indicateurs disponibles."
        ),
    }


# ============================================================
# HISTORIQUE SQLITE
# ============================================================

def init_signal_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal TEXT NOT NULL,
                entry REAL,
                stop_loss REAL,
                take_profit REAL,
                score REAL,
                rsi REAL
            )
        """)
        conn.commit()


def save_signal_record(result):
    if not result or not result.get("ok"):
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO signals (
                created_at, symbol, timeframe, signal,
                entry, stop_loss, take_profit, score, rsi
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            result["symbol"],
            result["timeframe"],
            result["signal"],
            result.get("entry"),
            result.get("stop_loss"),
            result.get("take_profit"),
            result.get("score"),
            result.get("rsi"),
        ))
        conn.commit()


def get_signal_history(limit=100):
    init_signal_db()

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT created_at, symbol, timeframe, signal,
                   entry, stop_loss, take_profit, score, rsi
            FROM signals
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
    )
    
# ============================================================
# TABLEAU DE BORD STREAMLIT
# ============================================================

init_signal_db()

st.title("ForexScope")
st.caption(
    "Analyse technique et structurelle — source : Yahoo Finance"
)

with st.sidebar:
    st.header("Paramètres")

    market_name = st.selectbox(
        "Marché",
        list(SYMBOLS.keys()),
        index=0,
    )

    timeframe = st.selectbox(
        "Unité de temps",
        list(TIMEFRAMES.keys()),
        index=2,
    )

    candle_count = st.slider(
        "Nombre de bougies",
        min_value=100,
        max_value=500,
        value=300,
        step=50,
    )

    st.caption(f"Symbole Yahoo : {SYMBOLS[market_name]}")
    st.caption("Les données peuvent différer des cotations de ton broker.")

    if st.button("Actualiser les données", use_container_width=True):
        fetch_candles.clear()
        st.rerun()


symbol = SYMBOLS[market_name]

status = get_connection_status(symbol)

if status["ok"]:
    st.success(status["message"])
else:
    st.warning(status["message"])

tab_analysis, tab_multi, tab_history, tab_settings = st.tabs([
    "Analyse",
    "Multi-unités de temps",
    "Historique",
    "Informations",
])


# ============================================================
# ONGLET ANALYSE
# ============================================================

with tab_analysis:
    st.subheader(f"{market_name} — {timeframe}")

    candles = fetch_candles(symbol, timeframe, candle_count)

    if candles.empty:
        st.error(
            "Aucune bougie disponible. Vérifie le symbole, "
            "l'intervalle et la disponibilité de Yahoo Finance."
        )
    else:
        df = add_indicators(candles)

        if df.empty:
            st.error("Les indicateurs n'ont pas pu être calculés.")
        else:
            last = df.iloc[-1]

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Dernière clôture",
                f"{last['close']:.5f}",
            )

            col2.metric(
                "RSI (14)",
                f"{last['rsi']:.2f}"
                if pd.notna(last["rsi"]) else "N/D",
            )

            col3.metric(
                "EMA 20",
                f"{last['ema20']:.5f}"
                if pd.notna(last["ema20"]) else "N/D",
            )

            col4.metric(
                "ATR (14)",
                f"{last['atr']:.5f}"
                if pd.notna(last["atr"]) else "N/D",
            )

            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df["time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Bougies",
            ))

            fig.add_trace(go.Scatter(
                x=df["time"],
                y=df["ema20"],
                mode="lines",
                name="EMA 20",
            ))

            fig.add_trace(go.Scatter(
                x=df["time"],
                y=df["ema50"],
                mode="lines",
                name="EMA 50",
            ))

            fig.update_layout(
                height=550,
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=30, b=10),
            )

            st.plotly_chart(fig, use_container_width=True)

            signal_result = generate_trading_signal(
                symbol, timeframe
            )

            st.subheader("Signal de trading")

            if not signal_result.get("ok"):
                st.warning(signal_result.get(
                    "message", "Signal indisponible."
                ))
            else:
                result = signal_result

                col1, col2, col3 = st.columns(3)

                col1.metric("Décision", result["signal"])
                col2.metric("Score technique", f'{result["score"]}/100')
                col3.metric("RSI", f'{result["rsi"]:.2f}')

                st.write(f"**Tendance :** {result['trend']}")
                st.write(f"**Structure :** {result['structure']}")
                st.write(f"**Liquidité :** {result['liquidity']}")
                st.write(f"**Sweep détecté :** {result['sweep'] or 'Aucun'}")

                if result["signal"] in ("Achat", "Vente"):
                    c1, c2, c3 = st.columns(3)

                    c1.metric("Entrée indicative", f'{result["entry"]:.5f}')
                    c2.metric("Stop Loss", f'{result["stop_loss"]:.5f}')
                    c3.metric("Take Profit", f'{result["take_profit"]:.5f}')

                    if st.button("Enregistrer ce signal"):
                        save_signal_record(result)
                        st.success("Signal enregistré dans l'historique.")
                else:
                    st.info(
                        "Les conditions ne sont pas suffisamment réunies. "
                        "Aucune entrée n'est proposée."
                    )

                st.caption(result["message"])


# ============================================================
# ONGLET MULTI-UNITES DE TEMPS
# ============================================================

with tab_multi:
    st.subheader("Analyse multi-unités de temps")

    if st.button("Lancer l'analyse multi-unités"):
        with st.spinner("Analyse des unités de temps..."):
            multi = multi_timeframe_analysis(symbol)

        if not multi.get("ok"):
            st.warning(multi["message"])
        else:
            st.metric("Consensus indicatif", multi["consensus"])
            st.metric(
                "Accord entre unités",
                f'{multi["agreement"]:.1f} %',
            )

            rows = []

            for tf, item in multi["timeframes"].items():
                if item.get("ok"):
                    rows.append({
                        "Unité": tf,
                        "Direction": item["direction"],
                        "Tendance": item["trend"],
                        "RSI": round(item["rsi"], 2),
                        "Structure": item["structure"],
                        "Sweep": item["sweep"] or "Aucun",
                    })
                else:
                    rows.append({
                        "Unité": tf,
                        "Direction": "N/D",
                        "Tendance": "N/D",
                        "RSI": "N/D",
                        "Structure": item["message"],
                        "Sweep": "N/D",
                    })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "L'accord entre unités mesure leur orientation commune ; "
                "ce n'est pas une probabilité de réussite."
            )


# ============================================================
# ONGLET HISTORIQUE
# ============================================================

with tab_history:
    st.subheader("Historique des signaux enregistrés")

    history = get_signal_history()

    if history.empty:
        st.info("Aucun signal enregistré pour le moment.")
    else:
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Télécharger l'historique CSV",
            data=csv_data,
            file_name="forexscope_historique.csv",
            mime="text/csv",
        )


# ============================================================
# ONGLET INFORMATIONS
# ============================================================

with tab_settings:
    st.subheader("Source et limites des données")

    st.write("**Fournisseur :** Yahoo Finance")
    st.write(f"**Symbole utilisé :** {symbol}")
    st.write(f"**Unité sélectionnée :** {timeframe}")
    st.write(f"**Bougies demandées :** {candle_count}")

    st.warning(
        "Yahoo Finance peut fournir des données retardées, "
        "incomplètes ou indisponibles selon le marché et l'intervalle. "
        "Le symbole Gold utilise des contrats à terme sur l'or (GC=F), "
        "pas nécessairement le prix spot XAU/USD de ton broker."
    )

    st.write(
        "Les signaux sont des indications techniques, pas des garanties "
        "de profit. Vérifie toujours les prix et les niveaux sur ton "
        "broker avant toute décision."
        )
    

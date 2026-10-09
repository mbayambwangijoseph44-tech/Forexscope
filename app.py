"""
deriv_client.py
Connexion à l'API Deriv (WebSocket) : récupération des bougies historiques
et des prix en direct, pour toutes les unités de temps utilisées par Forexscope.
"""

import asyncio
import json
import websockets
import pandas as pd
from datetime import datetime, timezone

DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"

# Correspondance unité de temps Forexscope -> granularité Deriv (en secondes)
GRANULARITES = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
    "MN": 2592000,
}

PAIRES_MAJEURES = {
    "EUR/USD": "frxEURUSD",
    "GBP/USD": "frxGBPUSD",
    "USD/JPY": "frxUSDJPY",
    "USD/CHF": "frxUSDCHF",
    "AUD/USD": "frxAUDUSD",
    "USD/CAD": "frxUSDCAD",
    "NZD/USD": "frxNZDUSD",
    "XAU/USD": "frxXAUUSD",
}


class DerivClient:
    """
    Client simple pour dialoguer avec l'API Deriv.
    Chaque appel ouvre sa propre connexion WebSocket (plus robuste pour un
    déploiement Streamlit où le script se relance souvent) plutôt que de
    garder une connexion persistante partagée entre les reruns.
    """

    def __init__(self, app_id: int = 1089):
        self.url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"

    async def _request(self, payload: dict, timeout: float = 15.0) -> dict:
        async with websockets.connect(self.url, open_timeout=timeout) as ws:
            await ws.send(json.dumps(payload))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                data = json.loads(raw)
                if data.get("msg_type") in ("candles", "history", "tick", "ticks_history"):
                    return data
                if "error" in data:
                    raise RuntimeError(f"Erreur API Deriv : {data['error'].get('message')}")
                # On ignore les autres messages (ping/pong, etc.) et on continue d'attendre
                if "candles" in data or "history" in data or "tick" in data:
                    return data

    async def get_candles_async(self, symbole: str, granularite_s: int, count: int = 300) -> pd.DataFrame:
        """
        Récupère `count` bougies historiques pour un symbole et une granularité donnés.
        Retourne un DataFrame avec colonnes : epoch, open, high, low, close, time (datetime).
        """
        payload = {
            "ticks_history": symbole,
            "style": "candles",
            "granularity": granularite_s,
            "count": count,
            "end": "latest",
        }
        data = await self._request(payload)

        candles = data.get("candles")
        if not candles:
            return pd.DataFrame(columns=["epoch", "open", "high", "low", "close", "time"])

        df = pd.DataFrame(candles)
        df = df.rename(columns={"epoch": "epoch"})
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
        df = df.sort_values("epoch").reset_index(drop=True)
        return df[["epoch", "open", "high", "low", "close", "time"]]

    def get_candles(self, symbole: str, granularite_s: int, count: int = 300) -> pd.DataFrame:
        """Version synchrone (pratique à appeler depuis Streamlit)."""
        return asyncio.run(self.get_candles_async(symbole, granularite_s, count))

    async def get_last_tick_async(self, symbole: str) -> dict:
        """Récupère le dernier prix connu (tick) pour un symbole, sans s'abonner en continu."""
        payload = {"ticks_history": symbole, "style": "ticks", "count": 1, "end": "latest"}
        data = await self._request(payload)
        prices = data.get("history", {}).get("prices", [])
        times = data.get("history", {}).get("times", [])
        if not prices:
            return {}
        return {"price": float(prices[-1]), "epoch": int(times[-1])}

    def get_last_tick(self, symbole: str) -> dict:
        return asyncio.run(self.get_last_tick_async(symbole))

    def marche_est_ouvert(self, symbole: str) -> bool:
        """
        Vérifie si des données récentes sont disponibles pour ce symbole.
        Si le dernier tick date de plus de 5 minutes, on considère le marché
        fermé ou les données insuffisantes (ex: weekend Forex).
        """
        try:
            tick = self.get_last_tick(symbole)
            if not tick:
                return False
            age_secondes = datetime.now(timezone.utc).timestamp() - tick["epoch"]
            return age_secondes < 300
        except Exception:
            return False
                    """
indicators.py
Indicateurs techniques calculés uniquement avec pandas/numpy
(pas de dépendance externe fragile type ta-lib — important pour un
déploiement simple sur Streamlit Cloud).
"""

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, periode: int = 14) -> pd.Series:
    """Average True Range — sert à la marge de sécurité du Stop Loss."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(periode).mean()


def rsi(df: pd.DataFrame, periode: int = 14) -> pd.Series:
    """Relative Strength Index — utilisé pour détecter les divergences."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    perte = -delta.clip(upper=0)
    moy_gain = gain.rolling(periode).mean()
    moy_perte = perte.rolling(periode).mean()
    rs = moy_gain / moy_perte.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame, rapide: int = 12, lent: int = 26, signal: int = 9):
    """MACD — ligne MACD, ligne de signal, et histogramme."""
    ema_rapide = df["close"].ewm(span=rapide, adjust=False).mean()
    ema_lent = df["close"].ewm(span=lent, adjust=False).mean()
    ligne_macd = ema_rapide - ema_lent
    ligne_signal = ligne_macd.ewm(span=signal, adjust=False).mean()
    histogramme = ligne_macd - ligne_signal
    return ligne_macd, ligne_signal, histogramme


def adx(df: pd.DataFrame, periode: int = 14) -> pd.Series:
    """
    Average Directional Index — mesure la force directionnelle du marché.
    Utilisé par le filtre de régime (Tendance si ADX >= seuil, sinon Range).
    """
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_lisse = tr.rolling(periode).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(periode).mean() / atr_lisse.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(periode).mean() / atr_lisse.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(periode).mean()


def moyenne_mobile(df: pd.DataFrame, periode: int) -> pd.Series:
    return df["close"].rolling(periode).mean()


def detecter_divergence(df: pd.DataFrame, colonne_momentum: pd.Series, fenetre: int = 20) -> str | None:
    """
    Détecte une divergence simple entre le prix et un indicateur de momentum
    (RSI ou MACD) sur les `fenetre` dernières bougies.
    Retourne "haussiere", "baissiere", ou None.
    """
    sous_ensemble = df.tail(fenetre).reset_index(drop=True)
    momentum = colonne_momentum.tail(fenetre).reset_index(drop=True)

    if len(sous_ensemble) < fenetre or momentum.isna().all():
        return None

    idx_prix_max = sous_ensemble["high"].idxmax()
    idx_prix_min = sous_ensemble["low"].idxmin()
    idx_mom_max = momentum.idxmax()
    idx_mom_min = momentum.idxmin()

    # Nouveau plus haut du prix vers la fin de la fenêtre, mais pas du momentum
    derniers_tiers = int(fenetre * 0.66)
    if idx_prix_max >= derniers_tiers and idx_mom_max < derniers_tiers:
        return "baissiere"  # épuisement haussier => divergence baissière
    if idx_prix_min >= derniers_tiers and idx_mom_min < derniers_tiers:
        return "haussiere"  # épuisement baissier => divergence haussière

    return None
    """
regime.py
Détermine si le marché est en RÉGIME DE TENDANCE ou en RÉGIME DE RANGE,
et calcule le biais de fond (haussier / baissier / neutre).

Règle (cf. prompt validé) :
ADX >= 23 ET amplitude des 5 dernières bougies > 1.15x la baseline
des 20 bougies précédentes => TREND, sinon RANGE.
"""

import pandas as pd
from indicators import adx, moyenne_mobile


def calculer_regime(df: pd.DataFrame, seuil_adx: float = 23.0, ratio_amplitude: float = 1.15) -> dict:
    """
    Retourne un dict :
    {
        "regime": "TREND" | "RANGE",
        "adx": valeur actuelle,
        "biais": "haussier" | "baissier" | "neutre",
    }
    """
    if len(df) < 30:
        return {"regime": "RANGE", "adx": None, "biais": "neutre"}

    adx_series = adx(df)
    adx_actuel = adx_series.iloc[-1]

    amplitude = (df["high"] - df["low"])
    amplitude_recente = amplitude.tail(5).mean()
    baseline = amplitude.iloc[-25:-5].mean()

    est_tendance = False
    if pd.notna(adx_actuel) and pd.notna(baseline) and baseline > 0:
        est_tendance = (adx_actuel >= seuil_adx) and (amplitude_recente > ratio_amplitude * baseline)

    mm20 = moyenne_mobile(df, 20).iloc[-1]
    mm50 = moyenne_mobile(df, 50).iloc[-1] if len(df) >= 50 else None
    prix_actuel = df["close"].iloc[-1]

    biais = "neutre"
    if mm50 is not None and pd.notna(mm20) and pd.notna(mm50):
        if prix_actuel > mm20 > mm50:
            biais = "haussier"
        elif prix_actuel < mm20 < mm50:
            biais = "baissier"

    return {
        "regime": "TREND" if est_tendance else "RANGE",
        "adx": round(float(adx_actuel), 1) if pd.notna(adx_actuel) else None,
        "biais": biais,
    }
    """
structure_engine.py
Détection de la structure de marché : swings (HH/HL/LH/LL), cassures de
structure (BOS / CHOCH-MSS), prises de liquidité (sweeps), Order Blocks
et Fair Value Gaps.

Toute la détection se fait sur les bougies DÉJÀ CLÔTURÉES (df.iloc[:-1] si la
dernière bougie du flux est encore en formation) — voir commentaire dans
decision_engine.py sur ce point, essentiel pour éviter le bruit du score live.
"""

import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Swing:
    index: int
    prix: float
    type: str  # "high" ou "low"


@dataclass
class Structure:
    swings: list = field(default_factory=list)
    sequence: str = ""          # ex: "HH-HL-HH" ou "LH-LL-LH"
    dernier_evenement: str | None = None   # "BOS" ou "CHOCH"
    direction: str | None = None           # "haussiere" ou "baissiere"
    niveau_invalidation: float | None = None   # swing utilisé comme invalidation


def detecter_swings(df: pd.DataFrame, lookback: int = 3) -> list[Swing]:
    """
    Détecte les pivots (swing highs / swing lows) par méthode fractale :
    un sommet est un `high` supérieur aux `lookback` bougies de chaque côté.
    """
    swings = []
    highs, lows = df["high"].values, df["low"].values
    n = len(df)

    for i in range(lookback, n - lookback):
        fenetre_high = highs[i - lookback:i + lookback + 1]
        fenetre_low = lows[i - lookback:i + lookback + 1]

        if highs[i] == fenetre_high.max() and highs[i] > fenetre_high[:lookback].max():
            swings.append(Swing(index=i, prix=highs[i], type="high"))
        if lows[i] == fenetre_low.min() and lows[i] < fenetre_low[:lookback].min():
            swings.append(Swing(index=i, prix=lows[i], type="low"))

    swings.sort(key=lambda s: s.index)
    return swings


def analyser_structure(df: pd.DataFrame, lookback: int = 3) -> Structure:
    """
    Construit la séquence HH/HL/LH/LL à partir des swings détectés, et
    identifie le dernier événement structurel (BOS = continuation,
    CHOCH/MSS = changement de caractère).
    """
    swings = detecter_swings(df, lookback)
    if len(swings) < 4:
        return Structure(swings=swings)

    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]

    etiquettes = []
    for i in range(1, len(highs)):
        etiquettes.append(("HH", highs[i]) if highs[i].prix > highs[i - 1].prix else ("LH", highs[i]))
    for i in range(1, len(lows)):
        etiquettes.append(("HL", lows[i]) if lows[i].prix > lows[i - 1].prix else ("LL", lows[i]))

    etiquettes.sort(key=lambda t: t[1].index)
    sequence = "-".join(e[0] for e in etiquettes[-5:])

    prix_actuel = df["close"].iloc[-1]
    dernier_swing_high = highs[-1] if highs else None
    dernier_swing_low = lows[-1] if lows else None

    dernier_evenement, direction, niveau_invalidation = None, None, None

    tendance_haussiere = len([e for e in etiquettes[-4:] if e[0] in ("HH", "HL")]) >= 2
    tendance_baissiere = len([e for e in etiquettes[-4:] if e[0] in ("LH", "LL")]) >= 2

    if dernier_swing_high and prix_actuel > dernier_swing_high.prix:
        if tendance_haussiere:
            dernier_evenement, direction = "BOS", "haussiere"
        elif tendance_baissiere:
            dernier_evenement, direction = "CHOCH", "haussiere"
        niveau_invalidation = dernier_swing_low.prix if dernier_swing_low else None

    elif dernier_swing_low and prix_actuel < dernier_swing_low.prix:
        if tendance_baissiere:
            dernier_evenement, direction = "BOS", "baissiere"
        elif tendance_haussiere:
            dernier_evenement, direction = "CHOCH", "baissiere"
        niveau_invalidation = dernier_swing_high.prix if dernier_swing_high else None

    return Structure(
        swings=swings,
        sequence=sequence,
        dernier_evenement=dernier_evenement,
        direction=direction,
        niveau_invalidation=niveau_invalidation,
    )


def detecter_sweep(df: pd.DataFrame, swings: list[Swing], tolerance_pips: float = 0.0) -> dict | None:
    """
    Un sweep (prise de liquidité) = une bougie dont la mèche dépasse un
    ancien swing high/low, mais dont la clôture revient À L'INTÉRIEUR —
    contrairement à une vraie cassure (BOS/CHOCH) où la clôture va au-delà.
    """
    if len(df) < 2 or not swings:
        return None

    derniere = df.iloc[-1]
    swings_highs = [s for s in swings if s.type == "high"]
    swings_lows = [s for s in swings if s.type == "low"]

    if swings_highs:
        dernier_high = swings_highs[-1]
        if derniere["high"] > dernier_high.prix + tolerance_pips and derniere["close"] < dernier_high.prix:
            return {"type": "SSL_SWEEP", "niveau": dernier_high.prix, "direction": "baissiere"}

    if swings_lows:
        dernier_low = swings_lows[-1]
        if derniere["low"] < dernier_low.prix - tolerance_pips and derniere["close"] > dernier_low.prix:
            return {"type": "BSL_SWEEP", "niveau": dernier_low.prix, "direction": "haussiere"}

    return None


def detecter_fvg(df: pd.DataFrame, nb_bougies_recentes: int = 15) -> list[dict]:
    """
    Fair Value Gap (déséquilibre 3 bougies) :
    - FVG haussier  : high(bougie[i-2]) < low(bougie[i])
    - FVG baissier  : low(bougie[i-2])  > high(bougie[i])
    Retourne la liste des FVG détectés sur les N dernières bougies, avec leur
    fraîcheur (nombre de bougies depuis leur formation) et s'ils ont déjà été
    comblés (mitigés) par le prix depuis.
    """
    fvgs = []
    sous_df = df.tail(nb_bougies_recentes + 2).reset_index(drop=True)

    for i in range(2, len(sous_df)):
        b0, b2 = sous_df.iloc[i - 2], sous_df.iloc[i]
        if b2["low"] > b0["high"]:
            zone = (b0["high"], b2["low"])
            mitige = (sous_df["low"].iloc[i + 1:] <= zone[1]).any() if i + 1 < len(sous_df) else False
            fvgs.append({"type": "haussier", "zone": zone, "fraicheur": len(sous_df) - 1 - i, "mitige": bool(mitige)})
        elif b2["high"] < b0["low"]:
            zone = (b2["high"], b0["low"])
            mitige = (sous_df["high"].iloc[i + 1:] >= zone[0]).any() if i + 1 < len(sous_df) else False
            fvgs.append({"type": "baissier", "zone": zone, "fraicheur": len(sous_df) - 1 - i, "mitige": bool(mitige)})

    return fvgs


def detecter_order_block(df: pd.DataFrame, structure: Structure) -> dict | None:
    """
    Order Block simplifié : la dernière bougie opposée au mouvement avant
    un BOS/CHOCH impulsif.
    - BOS/CHOCH haussier -> dernière bougie baissière avant la cassure = OB haussier
    - BOS/CHOCH baissier -> dernière bougie haussière avant la cassure = OB baissier
    """
    if not structure.dernier_evenement or len(df) < 5:
        return None

    recent = df.tail(10).reset_index(drop=True)

    if structure.direction == "haussiere":
        bougies_baissieres = recent[recent["close"] < recent["open"]]
        if bougies_baissieres.empty:
            return None
        derniere_baissiere = bougies_baissieres.iloc[-1]
        return {
            "type": "haussier",
            "zone": (derniere_baissiere["low"], derniere_baissiere["open"]),
            "mitige": False,
        }
    else:
        bougies_haussieres = recent[recent["close"] > recent["open"]]
        if bougies_haussieres.empty:
            return None
        derniere_haussiere = bougies_haussieres.iloc[-1]
        return {
            "type": "baissier",
            "zone": (derniere_haussiere["open"], derniere_haussiere["high"]),
            "mitige": False,
    }
    

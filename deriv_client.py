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

from async_utils import run_async

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
        # En-têtes imitant un vrai navigateur : Cloudflare (qui protège les
        # serveurs Deriv) bloque plus facilement les requêtes qui ressemblent
        # à un script/bot, surtout depuis une IP d'hébergeur cloud.
        async with websockets.connect(
            self.url,
            open_timeout=timeout,
            user_agent_header=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            origin="https://app.deriv.com",
        ) as ws:
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
        return run_async(self.get_candles_async(symbole, granularite_s, count))

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
        return run_async(self.get_last_tick_async(symbole))

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

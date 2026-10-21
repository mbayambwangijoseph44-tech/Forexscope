"""
browser_bridge.py

Contourne le blocage Cloudflare observé côté serveur (HTTP 520 sur
Streamlit Cloud ET sur Render — donc lié aux IP de datacenter en général,
pas à une plateforme précise).

Principe : au lieu d'ouvrir la connexion WebSocket depuis le serveur Python
(IP cloud, souvent bloquée par Cloudflare), on l'ouvre depuis le NAVIGATEUR
de l'utilisateur (IP résidentielle/mobile, jamais bloquée), via un petit
script JavaScript exécuté côté client grâce à `streamlit-javascript`.
Le résultat est ensuite renvoyé à Python sous forme de texte JSON.

C'est très probablement la méthode utilisée par les plateformes no-code
(Bolt, Lovable, MeDo) qui fonctionnent sans ce problème.
"""

import json

from streamlit_javascript import st_javascript


def _js_requete_deriv(payload: dict, timeout_ms: int = 12000) -> str:
    """
    Construit le script JS qui ouvre une connexion WebSocket vers Deriv
    depuis le navigateur, envoie `payload`, attend la réponse pertinente,
    puis referme la connexion et renvoie le résultat en JSON (stringifié).
    """
    payload_json = json.dumps(payload)
    return f"""
    await new Promise((resolve) => {{
        let fini = false;
        const ws = new WebSocket("wss://ws.derivws.com/websockets/v3?app_id=1089");

        const terminer = (valeur) => {{
            if (fini) return;
            fini = true;
            try {{ ws.close(); }} catch (e) {{}}
            resolve(JSON.stringify(valeur));
        }};

        ws.onopen = () => {{
            ws.send(JSON.stringify({payload_json}));
        }};

        ws.onmessage = (event) => {{
            const data = JSON.parse(event.data);
            if (data.error) {{
                terminer({{ erreur: data.error.message || "Erreur inconnue" }});
            }} else if (data.candles || data.history || data.tick) {{
                terminer(data);
            }}
        }};

        ws.onerror = () => terminer({{ erreur: "Erreur de connexion WebSocket (navigateur)" }});
        setTimeout(() => terminer({{ erreur: "Délai dépassé (pas de réponse de Deriv)" }}), {timeout_ms});
    }});
    """


def get_last_tick_browser(symbole: str) -> dict:
    """Récupère le dernier prix connu, via une connexion ouverte par le navigateur."""
    payload = {"ticks_history": symbole, "style": "ticks", "count": 1, "end": "latest"}
    resultat_brut = st_javascript(_js_requete_deriv(payload))

    if not resultat_brut:
        return {}

    data = json.loads(resultat_brut)
    if "erreur" in data:
        raise RuntimeError(data["erreur"])

    prices = data.get("history", {}).get("prices", [])
    times = data.get("history", {}).get("times", [])
    if not prices:
        return {}
    return {"price": float(prices[-1]), "epoch": int(times[-1])}


def get_candles_browser(symbole: str, granularite_s: int, count: int = 50) -> list[dict]:
    """Récupère des bougies historiques, via une connexion ouverte par le navigateur."""
    payload = {
        "ticks_history": symbole,
        "style": "candles",
        "granularity": granularite_s,
        "count": count,
        "end": "latest",
    }
    resultat_brut = st_javascript(_js_requete_deriv(payload))

    if not resultat_brut:
        return []

    data = json.loads(resultat_brut)
    if "erreur" in data:
        raise RuntimeError(data["erreur"])

    return data.get("candles", [])
  

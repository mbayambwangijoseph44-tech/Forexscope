"""
browser_bridge.py (version SANS dépendance externe)

Contourne le blocage Cloudflare côté serveur (HTTP 520, confirmé sur
Streamlit Cloud ET Render) en ouvrant la connexion WebSocket vers Deriv
depuis le NAVIGATEUR de l'utilisateur plutôt que depuis le serveur Python.

Contrairement à la version précédente, celle-ci n'utilise AUCUN paquet
externe (plus de "streamlit-javascript") — seulement
`streamlit.components.v1.html`, qui fait partie du cœur de Streamlit et
ne peut donc pas échouer à l'installation.

Principe de communication navigateur -> Python (puisque components.html
est à sens unique) :
Le script JS, une fois la donnée reçue de Deriv, encode le résultat et
modifie l'URL de la page (paramètre `?fs_data=...`). Cela déclenche un
rechargement de Streamlit, qui lit la donnée via `st.query_params`.
"""

import base64
import json

import streamlit as st
import streamlit.components.v1 as components


def _lancer_requete_navigateur(payload: dict, cle_resultat: str, hauteur: int = 0):
    """
    Injecte un script JS invisible qui ouvre une connexion WebSocket vers
    Deriv depuis le navigateur, envoie `payload`, puis renvoie le résultat
    à Streamlit via les paramètres d'URL (clé `cle_resultat`).
    """
    payload_json = json.dumps(payload)

    composant_html = f"""
    <div id="statut-{cle_resultat}" style="font-family: sans-serif; font-size: 13px; color: #888;">
        Connexion à Deriv en cours (depuis votre navigateur)...
    </div>
    <script>
    (function() {{
        const ws = new WebSocket("wss://ws.derivws.com/websockets/v3?app_id=1089");
        const statutEl = document.getElementById("statut-{cle_resultat}");
        let fini = false;

        function envoyerResultat(valeur) {{
            if (fini) return;
            fini = true;
            try {{ ws.close(); }} catch (e) {{}}

            const encode = btoa(unescape(encodeURIComponent(JSON.stringify(valeur))));
            const url = new URL(window.parent.location.href);
            url.searchParams.set("{cle_resultat}", encode);
            url.searchParams.set("{cle_resultat}_ts", Date.now().toString());
            window.parent.location.href = url.toString();
        }}

        ws.onopen = () => {{
            statutEl.textContent = "Connecté — envoi de la requête...";
            ws.send(JSON.stringify({payload_json}));
        }};

        ws.onmessage = (event) => {{
            const data = JSON.parse(event.data);
            if (data.error) {{
                envoyerResultat({{ erreur: data.error.message || "Erreur inconnue" }});
            }} else if (data.candles || data.history || data.tick) {{
                statutEl.textContent = "Donnée reçue, transmission à l'application...";
                envoyerResultat(data);
            }}
        }};

        ws.onerror = () => envoyerResultat({{ erreur: "Erreur de connexion WebSocket (navigateur)" }});
        setTimeout(() => envoyerResultat({{ erreur: "Délai dépassé (pas de réponse de Deriv)" }}), 12000);
    }})();
    </script>
    """
    components.html(composant_html, height=hauteur if hauteur else 30)


def _lire_resultat(cle_resultat: str) -> dict | None:
    """Lit un résultat précédemment transmis par le navigateur via st.query_params."""
    params = st.query_params
    if cle_resultat not in params:
        return None
    try:
        brut = base64.b64decode(params[cle_resultat]).decode("utf-8")
        return json.loads(brut)
    except Exception:
        return None


def demander_dernier_tick(symbole: str):
    """
    Lance la requête du dernier prix depuis le navigateur.
    À appeler, PUIS relire le résultat avec `lire_dernier_tick()` après
    le rechargement automatique de la page (déclenché par le script JS).
    """
    payload = {"ticks_history": symbole, "style": "ticks", "count": 1, "end": "latest"}
    _lancer_requete_navigateur(payload, "fs_tick")


def lire_dernier_tick() -> dict:
    data = _lire_resultat("fs_tick")
    if not data:
        return {}
    if "erreur" in data:
        raise RuntimeError(data["erreur"])
    prices = data.get("history", {}).get("prices", [])
    times = data.get("history", {}).get("times", [])
    if not prices:
        return {}
    return {"price": float(prices[-1]), "epoch": int(times[-1])}


def demander_bougies(symbole: str, granularite_s: int, count: int = 50):
    payload = {
        "ticks_history": symbole,
        "style": "candles",
        "granularity": granularite_s,
        "count": count,
        "end": "latest",
    }
    _lancer_requete_navigateur(payload, "fs_candles")


def lire_bougies() -> list:
    data = _lire_resultat("fs_candles")
    if not data:
        return []
    if "erreur" in data:
        raise RuntimeError(data["erreur"])
    return data.get("candles", [])

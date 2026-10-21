"""
app.py — ÉTAPE 0 : validation de la connexion Deriv uniquement.

Objectif unique de cette version : prouver, de façon fiable et répétable
sur Streamlit Cloud, que la connexion à l'API Deriv fonctionne (prix en
direct + bougies historiques). Rien d'autre n'est ajouté tant que cette
étape n'est pas validée plusieurs fois de suite sans erreur.

Si une erreur survient, elle est affichée en entier (type d'exception +
message complet), jamais masquée par un message générique.
"""

import traceback

import streamlit as st

from deriv_client import DerivClient, PAIRES_MAJEURES, GRANULARITES
from browser_bridge import get_last_tick_browser, get_candles_browser

st.set_page_config(page_title="Forexscope — Diagnostic Deriv", page_icon="🔌", layout="centered")

st.title("🔌 Forexscope — Diagnostic de connexion Deriv")
st.caption(
    "Cette page ne fait qu'une chose : vérifier que la connexion à l'API Deriv "
    "fonctionne de façon fiable. Aucune analyse, aucun signal, aucune persistance "
    "pour l'instant — volontairement."
)

st.error(
    "⚠️ La méthode SERVEUR (ci-dessous, section 1) échoue actuellement avec "
    "HTTP 520 — confirmé sur Streamlit Cloud ET Render. Utilisez la section "
    "MÉTHODE NAVIGATEUR plus bas, qui contourne ce blocage.",
    icon="⚠️",
)

client = DerivClient()

paire_label = st.selectbox("Paire à tester", list(PAIRES_MAJEURES.keys()), index=0)
symbole = PAIRES_MAJEURES[paire_label]

unite_label = st.selectbox("Unité de temps à tester", list(GRANULARITES.keys()), index=4)  # H1 par défaut
granularite = GRANULARITES[unite_label]

col1, col2 = st.columns(2)

with col1:
    if st.button("1️⃣ Tester le dernier prix (tick)", use_container_width=True):
        with st.spinner(f"Connexion à Deriv pour récupérer le dernier prix de {paire_label}..."):
            try:
                tick = client.get_last_tick(symbole)
                if tick:
                    st.success("Connexion réussie.")
                    st.json(tick)
                else:
                    st.warning(
                        "Connexion établie, mais aucune donnée renvoyée. "
                        "Le marché est peut-être fermé pour ce symbole."
                    )
            except Exception as e:
                st.error(f"Échec de connexion : {type(e).__name__}")
                st.code(f"{e}\n\n{traceback.format_exc()}", language="text")

with col2:
    if st.button("2️⃣ Tester les bougies historiques", use_container_width=True):
        with st.spinner(f"Récupération des bougies {unite_label} pour {paire_label}..."):
            try:
                df = client.get_candles(symbole, granularite, count=50)
                if df.empty:
                    st.warning("Connexion établie, mais aucune bougie renvoyée.")
                else:
                    st.success(f"{len(df)} bougies récupérées avec succès.")
                    st.dataframe(df.tail(10), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Échec de connexion : {type(e).__name__}")
                st.code(f"{e}\n\n{traceback.format_exc()}", language="text")

st.divider()

if st.button("3️⃣ Vérifier l'état du marché (ouvert / fermé)", use_container_width=True):
    with st.spinner("Vérification..."):
        try:
            ouvert = client.marche_est_ouvert(symbole)
            if ouvert:
                st.success(f"Le marché {paire_label} semble actif (données récentes de moins de 5 min).")
            else:
                st.warning(f"Le marché {paire_label} semble fermé ou les données sont obsolètes.")
        except Exception as e:
            st.error(f"Échec de connexion : {type(e).__name__}")
            st.code(f"{e}\n\n{traceback.format_exc()}", language="text")

st.divider()
st.subheader("🌐 Méthode NAVIGATEUR (contourne le blocage Cloudflare)")
st.caption(
    "Ici, la connexion WebSocket est ouverte depuis le navigateur de "
    "l'utilisateur (votre téléphone/ordinateur), pas depuis le serveur. "
    "L'IP qui contacte Deriv est donc résidentielle/mobile, pas une IP de "
    "datacenter — ce que Cloudflare bloque beaucoup moins facilement."
)

col3, col4 = st.columns(2)

with col3:
    if st.button("1️⃣🌐 Tester le dernier prix (navigateur)", use_container_width=True):
        with st.spinner("Connexion depuis votre navigateur vers Deriv..."):
            try:
                tick = get_last_tick_browser(symbole)
                if tick:
                    st.success("Connexion réussie depuis le navigateur !")
                    st.json(tick)
                elif tick == {}:
                    st.info("En attente de la réponse du navigateur... clique à nouveau si rien ne s'affiche.")
            except Exception as e:
                st.error(f"Échec (navigateur) : {type(e).__name__}")
                st.code(str(e), language="text")

with col4:
    if st.button("2️⃣🌐 Tester les bougies (navigateur)", use_container_width=True):
        with st.spinner("Connexion depuis votre navigateur vers Deriv..."):
            try:
                candles = get_candles_browser(symbole, granularite, count=50)
                if candles:
                    st.success(f"{len(candles)} bougies récupérées depuis le navigateur !")
                    st.dataframe(candles[-10:], use_container_width=True, hide_index=True)
                else:
                    st.info("En attente de la réponse du navigateur... clique à nouveau si rien ne s'affiche.")
            except Exception as e:
                st.error(f"Échec (navigateur) : {type(e).__name__}")
                st.code(str(e), language="text")

st.divider()
st.caption(
    "Prochaine étape UNE FOIS que la méthode navigateur passe de façon répétée et fiable : "
    "brancher le moteur d'analyse (structure, régime, scores) sur ces données (Partie 2)."
)

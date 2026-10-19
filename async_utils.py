"""
async_utils.py
Streamlit ne gère pas nativement asyncio de façon homogène selon
l'environnement (local vs Streamlit Cloud, versions de Python). Ce module
centralise une façon robuste d'exécuter une coroutine depuis du code
synchrone (Streamlit), pour éviter les erreurs classiques :
- "RuntimeError: There is no current event loop"
- "RuntimeError: asyncio.run() cannot be called from a running event loop"
- "RuntimeError: Event loop is closed"

Principe : ne JAMAIS réutiliser une boucle d'un rerun précédent. On crée une
boucle neuve, on l'exécute jusqu'au bout, on la ferme explicitement.
"""

import asyncio


def run_async(coroutine):
    """
    Exécute une coroutine de façon sûre, quel que soit l'état d'asyncio
    dans le thread courant. À utiliser pour CHAQUE appel réseau Deriv
    depuis Streamlit plutôt que d'appeler asyncio.run() directement.
    """
    try:
        # Cas normal : aucune boucle active dans ce thread -> le plus simple fonctionne.
        return asyncio.run(coroutine)
    except RuntimeError as e:
        if "cannot be called from a running event loop" not in str(e):
            raise
        # Cas rare (certains environnements Streamlit Cloud / notebooks) :
        # une boucle tourne déjà dans ce thread. On en crée une nouvelle,
        # isolée, dédiée à cet appel, puis on la referme proprement.
        nouvelle_boucle = asyncio.new_event_loop()
        try:
            return nouvelle_boucle.run_until_complete(coroutine)
        finally:
            nouvelle_boucle.close()
  

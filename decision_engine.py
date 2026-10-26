"""
decision_engine.py
Classifie chaque setup (CONTINUATION / RETOURNEMENT), calcule les trois
scores (Anticipation, Timing, Confirmation) avec la pondération validée
(40 / 35 / 25, pénalité si Confirmation domine seule), et détermine l'état
du signal (ATTENTE / PRÉ-SIGNAL / CONFIRMATION / SIGNAL).

IMPORTANT (règle validée à plusieurs reprises dans la conception) :
Toute la détection de structure utilisée ici doit se baser sur des bougies
DÉJÀ CLÔTURÉES. Si le flux de données contient la bougie en cours de
formation, il faut la retirer avant d'appeler `analyser()`
(ex: df.iloc[:-1] si on sait que la dernière ligne est encore ouverte) —
sinon le score peut fluctuer à chaque tick, recréant le bruit qu'on a
cherché à éliminer sur l'affichage du signal figé.
"""

from dataclasses import dataclass, field

import pandas as pd

from indicators import rsi, macd, atr, detecter_divergence
from regime import calculer_regime
from structure_engine import analyser_structure, detecter_sweep, detecter_fvg, detecter_order_block


@dataclass
class ResultatAnalyse:
    regime: str = "RANGE"
    biais: str = "neutre"
    type_setup: str | None = None          # "CONTINUATION" | "RETOURNEMENT" | None
    direction: str | None = None           # "haussiere" | "baissiere"
    score_anticipation: float = 0.0
    score_timing: float = 0.0
    score_confirmation: float = 0.0
    score_global: float = 0.0
    etat: str = "ATTENTE"                  # ATTENTE | PRE_SIGNAL | CONFIRMATION | SIGNAL
    niveau_invalidation: float | None = None
    zone_entree: tuple | None = None       # (bas, haut) si entrée sur zone
    sweep: dict | None = None
    fvg_pertinents: list = field(default_factory=list)
    order_block: dict | None = None
    raisons_favorables: list = field(default_factory=list)
    raisons_defavorables: list = field(default_factory=list)
    penalites: list = field(default_factory=list)


# Seuils de décision (modifiables depuis les paramètres utilisateur plus tard)
SEUIL_SIGNAL = 70
SEUIL_CONFIRMATION = 55
SEUIL_PRE_SIGNAL = 40


def _retirer_bougie_en_formation(df: pd.DataFrame, derniere_est_ouverte: bool) -> pd.DataFrame:
    """Règle critique : ne jamais baser le scoring sur une bougie encore en formation."""
    if derniere_est_ouverte and len(df) > 1:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def analyser(
    df: pd.DataFrame,
    df_contexte: pd.DataFrame | None = None,
    derniere_est_ouverte: bool = True,
) -> ResultatAnalyse:
    """
    Analyse complète d'une unité de temps : régime, structure, classification
    du setup, scoring pondéré, et état de décision.

    `df` : bougies de l'unité d'ENTRÉE (ex: H4)
    `df_contexte` : bougies de l'unité SUPÉRIEURE (ex: D1), optionnel mais
                    recommandé pour le filtre de confluence.
    """
    df = _retirer_bougie_en_formation(df, derniere_est_ouverte)
    resultat = ResultatAnalyse()

    if len(df) < 30:
        resultat.raisons_defavorables.append("Pas assez de données pour une analyse fiable.")
        return resultat

    # --- 1. Régime de marché ---
    regime_info = calculer_regime(df)
    resultat.regime = regime_info["regime"]
    resultat.biais = regime_info["biais"]

    # --- 2. Structure ---
    structure = analyser_structure(df)
    resultat.niveau_invalidation = structure.niveau_invalidation

    if not structure.dernier_evenement:
        resultat.raisons_defavorables.append("Aucune cassure de structure récente identifiée.")
        return resultat

    # --- 3. Classification CONTINUATION vs RETOURNEMENT ---
    if structure.dernier_evenement == "BOS" and resultat.regime == "TREND":
        resultat.type_setup = "CONTINUATION"
        resultat.direction = structure.direction
    elif structure.dernier_evenement == "CHOCH":
        resultat.type_setup = "RETOURNEMENT"
        resultat.direction = structure.direction
    else:
        # Cas ambigu (ex: BOS en régime RANGE) -> pas d'étiquette forcée
        resultat.raisons_defavorables.append(
            f"Setup ambigu ({structure.dernier_evenement} en régime {resultat.regime}) — non qualifié."
        )
        return resultat

    # --- 4. Liquidité et zones ---
    sweep = detecter_sweep(df, structure.swings)
    resultat.sweep = sweep
    fvgs = detecter_fvg(df)
    resultat.fvg_pertinents = [f for f in fvgs if not f["mitige"]]
    ob = detecter_order_block(df, structure)
    resultat.order_block = ob

    # --- 5. Indicateurs de momentum ---
    rsi_series = rsi(df)
    _, _, macd_hist = macd(df)
    divergence = detecter_divergence(df, rsi_series)
    atr_actuel = atr(df).iloc[-1]

    # --- 6. Scoring différencié selon le type de setup ---
    score_anticipation = 0.0
    score_timing = 0.0
    score_confirmation = 0.0

    if resultat.type_setup == "CONTINUATION":
        momentum_accelere = abs(macd_hist.iloc[-1]) > abs(macd_hist.iloc[-3]) if len(macd_hist) > 3 else False
        if momentum_accelere:
            score_anticipation += 25
            resultat.raisons_favorables.append("Momentum encore présent dans le sens de la tendance.")
        if resultat.order_block or resultat.fvg_pertinents:
            score_anticipation += 20
            resultat.raisons_favorables.append("Pullback vers une zone OB/FVG identifiée.")
        if divergence and divergence != resultat.direction:
            score_anticipation -= 10
            resultat.penalites.append("Divergence contradictoire détectée sur un setup de continuation.")

        score_confirmation = 40 if structure.dernier_evenement == "BOS" else 15

    elif resultat.type_setup == "RETOURNEMENT":
        if divergence == resultat.direction:
            score_anticipation += 20
            resultat.raisons_favorables.append(f"Divergence {divergence} confirmée.")
        if sweep:
            score_anticipation += 20
            resultat.raisons_favorables.append(f"Liquidity sweep détecté ({sweep['type']}).")
            resultat.zone_entree = (sweep["niveau"], sweep["niveau"])
        else:
            resultat.penalites.append("Retournement sans sweep de liquidité — moins fiable.")

        if resultat.order_block:
            score_anticipation += 10
            resultat.zone_entree = resultat.order_block["zone"]

        score_confirmation = 35 if sweep else 15

    # Confluence avec l'unité supérieure (filtre secondaire)
    confluence_ok = True
    if df_contexte is not None and len(df_contexte) >= 30:
        regime_contexte = calculer_regime(df_contexte)
        if regime_contexte["biais"] != "neutre" and regime_contexte["biais"] != resultat.direction:
            confluence_ok = False
            resultat.penalites.append("Contredit le biais de l'unité supérieure (contexte).")

    score_timing = 50 if confluence_ok else 20
    if resultat.zone_entree:
        score_timing += 20

    # Pénalité : confirmation forte SEULE, sans anticipation/timing suffisants
    if score_confirmation >= 35 and (score_anticipation < 25 or score_timing < 35):
        score_confirmation -= 15
        resultat.penalites.append(
            "Confirmation forte mais Anticipation/Timing insuffisants — mouvement probablement déjà mature."
        )

    resultat.score_anticipation = max(0, min(100, score_anticipation * 2))  # mise à l'échelle 0-100
    resultat.score_timing = max(0, min(100, score_timing * 1.5))
    resultat.score_confirmation = max(0, min(100, score_confirmation * 1.8))

    # --- 7. Score global pondéré : 40% anticipation + 35% timing + 25% confirmation ---
    resultat.score_global = (
        0.40 * resultat.score_anticipation
        + 0.35 * resultat.score_timing
        + 0.25 * resultat.score_confirmation
    )

    # --- 8. État de décision ---
    if resultat.score_global >= SEUIL_SIGNAL and confluence_ok:
        resultat.etat = "SIGNAL"
    elif resultat.score_global >= SEUIL_CONFIRMATION:
        resultat.etat = "CONFIRMATION"
    elif resultat.score_global >= SEUIL_PRE_SIGNAL:
        resultat.etat = "PRE_SIGNAL"
    else:
        resultat.etat = "ATTENTE"

    return resultat

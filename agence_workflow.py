"""L'Agence IA du Bon Prix — orchestration multi-agents (LangGraph) du capstone Jour 3.

Ce module réutilise TELS QUELS les agents déjà construits pendant le cours :
- agents.specialist_agent.SpecialistAgent  (Jour 1 : modèle fine-tuné, déployé sur Modal)
- agents.frontier_agent.FrontierAgent      (Jour 3 : RAG + LLM frontier)

et les orchestre via un graphe LangGraph (Jour 2) pour produire, à partir d'une simple description
produit, une recommandation métier complète : prix juste estimé, verdict ("bonne affaire" / "prix
correct" / "trop cher"), justification et prix de revente suggéré.

Choix d'architecture : le routage est ICI déterministe (des arêtes fixes, pas un superviseur LLM),
contrairement aux exemples du Jour 2. Quand l'enchaînement des étapes ne dépend jamais du contenu —
comme ici : comparables → spécialiste → frontier → conseiller, toujours dans cet ordre — un routage
piloté par LLM n'apporterait ni fiabilité ni rapidité supplémentaires, juste de la latence et un coût
inutiles. Le superviseur LLM (vu au Jour 2) garde tout son intérêt quand la PROCHAINE étape dépend
réellement du contenu produit par les étapes précédentes.

La résilience, elle, est réelle : si le service Modal (spécialiste) ou l'API OpenAI (frontier) est
indisponible, le nœud correspondant renvoie une erreur capturée plutôt que de faire échouer tout le
graphe, et le Conseiller final recalcule sa recommandation avec les informations qui restent.
"""

import os
from typing import Annotated, Literal, Optional, TypedDict
import operator

import chromadb
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agents.frontier_agent import FrontierAgent
from agents.specialist_agent import SpecialistAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "products_vectorstore")


# -----------------------------------------------------------------------------
# Initialisation paresseuse des ressources lourdes (un seul chargement par
# conteneur/processus, pas à l'import du module).
# -----------------------------------------------------------------------------
_collection = None
_frontier_agent = None
_specialist_agent = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_PATH)
        _collection = client.get_or_create_collection("products")
    return _collection


def get_frontier_agent() -> FrontierAgent:
    global _frontier_agent
    if _frontier_agent is None:
        _frontier_agent = FrontierAgent(get_collection())
    return _frontier_agent


def get_specialist_agent() -> SpecialistAgent:
    global _specialist_agent
    if _specialist_agent is None:
        _specialist_agent = SpecialistAgent()
    return _specialist_agent


# -----------------------------------------------------------------------------
# État partagé
# -----------------------------------------------------------------------------
class AgencyState(TypedDict):
    description: str
    asking_price: Optional[float]
    comparables: list
    specialist_price: Optional[float]
    specialist_error: Optional[str]
    frontier_price: Optional[float]
    frontier_error: Optional[str]
    fair_price: Optional[float]
    verdict: str
    justification: str
    suggested_resale_price: Optional[float]
    messages: Annotated[list, operator.add]


# -----------------------------------------------------------------------------
# Nœuds
# -----------------------------------------------------------------------------
def comparables_node(state: AgencyState) -> dict:
    """Recherche RAG (via FrontierAgent.find_similars) : produits similaires + leurs prix."""
    try:
        agent = get_frontier_agent()
        documents, prices = agent.find_similars(state["description"])
        comparables = [{"description": doc[:200], "price": price} for doc, price in zip(documents, prices)]
    except Exception as e:
        comparables = []
        return {"comparables": comparables, "messages": [{"agent": "Comparables", "content": f"Recherche indisponible : {e}"}]}
    return {"comparables": comparables, "messages": [{"agent": "Comparables", "content": f"{len(comparables)} produits similaires trouvés."}]}


def specialiste_node(state: AgencyState) -> dict:
    """Estimation du modèle fine-tuné (Jour 1), déployé sur Modal (Jour 3)."""
    try:
        agent = get_specialist_agent()
        price = agent.price(state["description"])
        return {"specialist_price": price, "specialist_error": None, "messages": [{"agent": "Spécialiste", "content": f"${price:.2f}"}]}
    except Exception as e:
        return {"specialist_price": None, "specialist_error": str(e), "messages": [{"agent": "Spécialiste", "content": f"Indisponible : {e}"}]}


def frontier_node(state: AgencyState) -> dict:
    """Estimation RAG + LLM frontier (Jour 3)."""
    try:
        agent = get_frontier_agent()
        price = agent.price(state["description"])
        return {"frontier_price": price, "frontier_error": None, "messages": [{"agent": "Frontier", "content": f"${price:.2f}"}]}
    except Exception as e:
        return {"frontier_price": None, "frontier_error": str(e), "messages": [{"agent": "Frontier", "content": f"Indisponible : {e}"}]}


class Recommandation(BaseModel):
    verdict: Literal["Bonne affaire", "Prix correct", "Trop cher", "Estimation incertaine"]
    justification: str
    suggested_resale_price: float


CONSEILLER_PROMPT = """Tu es le Conseiller de l'Agence IA du Bon Prix. Tu reçois une estimation de
"prix juste" pour un produit, éventuellement un prix demandé par le vendeur, et des produits
comparables issus d'un catalogue réel.

Détermine le verdict :
- "Bonne affaire" si le prix demandé est nettement inférieur (>10%) au prix juste estimé.
- "Prix correct" si le prix demandé est proche du prix juste estimé (± 10%), ou si aucun prix demandé
  n'est fourni (concentre alors ta justification sur l'estimation elle-même).
- "Trop cher" si le prix demandé dépasse nettement (>10%) le prix juste estimé.
- "Estimation incertaine" si aucune estimation fiable n'a pu être calculée.

Propose aussi un prix de revente suggéré (nombre, en dollars). Justifie en 3-4 phrases factuelles."""


def conseiller_node(state: AgencyState) -> dict:
    """Combine les estimations disponibles et produit la recommandation métier finale."""
    specialist = state.get("specialist_price")
    frontier = state.get("frontier_price")

    # Pondération : 65% Frontier (RAG, généralement le plus fiable) / 35% Spécialiste, quand les deux
    # sont disponibles ; sinon on utilise celui qui reste ; sinon aucune estimation n'est possible.
    if specialist is not None and frontier is not None:
        fair_price = round(frontier * 0.65 + specialist * 0.35, 2)
    elif frontier is not None:
        fair_price = round(frontier, 2)
    elif specialist is not None:
        fair_price = round(specialist, 2)
    else:
        fair_price = None

    comparables_txt = "\n".join(
        f"- {c['description']} : ${c['price']:.2f}" for c in state.get("comparables", [])[:5]
    ) or "(aucun comparable trouvé)"

    context = (
        f"Description du produit : {state['description']}\n"
        f"Prix demandé par le vendeur : {state.get('asking_price', 'non fourni')}\n"
        f"Estimation du modèle spécialiste (fine-tuné) : {specialist if specialist is not None else 'indisponible'}\n"
        f"Estimation du modèle frontier (RAG + LLM) : {frontier if frontier is not None else 'indisponible'}\n"
        f"Prix juste combiné : {fair_price if fair_price is not None else 'indisponible'}\n"
        f"Produits comparables :\n{comparables_txt}"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    decision: Recommandation = llm.with_structured_output(Recommandation).invoke([
        SystemMessage(content=CONSEILLER_PROMPT),
        HumanMessage(content=context),
    ])

    return {
        "fair_price": fair_price,
        "verdict": decision.verdict,
        "justification": decision.justification,
        "suggested_resale_price": decision.suggested_resale_price,
        "messages": [{"agent": "Conseiller", "content": decision.justification}],
    }


# -----------------------------------------------------------------------------
# Construction du graphe (routage déterministe, cf. docstring du module)
# -----------------------------------------------------------------------------
_graph = None


def build_graph():
    workflow = StateGraph(AgencyState)
    workflow.add_node("Comparables", comparables_node)
    workflow.add_node("Spécialiste", specialiste_node)
    workflow.add_node("Frontier", frontier_node)
    workflow.add_node("Conseiller", conseiller_node)

    workflow.set_entry_point("Comparables")
    workflow.add_edge("Comparables", "Spécialiste")
    workflow.add_edge("Spécialiste", "Frontier")
    workflow.add_edge("Frontier", "Conseiller")
    workflow.add_edge("Conseiller", END)

    return workflow.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_estimation(description: str, asking_price: Optional[float] = None) -> dict:
    """Point d'entrée unique : lance le graphe et renvoie un dict prêt à sérialiser en JSON.

    Utilisé à la fois par le notebook (démonstration locale) et par le point de terminaison Modal
    (agence_modal_app.py) : une seule source de vérité pour la logique métier.
    """
    graph = get_graph()
    initial_state: AgencyState = {
        "description": description,
        "asking_price": asking_price,
        "comparables": [],
        "specialist_price": None,
        "specialist_error": None,
        "frontier_price": None,
        "frontier_error": None,
        "fair_price": None,
        "verdict": "",
        "justification": "",
        "suggested_resale_price": None,
        "messages": [],
    }
    result = graph.invoke(initial_state)

    return {
        "description": description,
        "asking_price": asking_price,
        "fair_price": result.get("fair_price"),
        "specialist_price": result.get("specialist_price"),
        "frontier_price": result.get("frontier_price"),
        "verdict": result.get("verdict"),
        "justification": result.get("justification"),
        "suggested_resale_price": result.get("suggested_resale_price"),
        "comparables": result.get("comparables", [])[:5],
        "trace": [
            {"agent": m["agent"], "content": m["content"]} for m in result.get("messages", [])
        ],
    }


if __name__ == "__main__":
    import json

    demo = run_estimation(
        "iPhone 12, 64GB, débloqué tout opérateur, bon état, quelques micro-rayures au dos",
        asking_price=250.0,
    )
    print(json.dumps(demo, indent=2, ensure_ascii=False))

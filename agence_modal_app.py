"""L'Agence IA du Bon Prix — déploiement Modal (API REST).

Déploiement :
    modal deploy agence_modal_app.py

Secret requis (à créer une seule fois si vous ne l'avez pas déjà) :
    modal secret create openai-secret OPENAI_API_KEY=sk-votre-cle

Le secret "huggingface-secret" n'est PAS requis ici : il est déjà utilisé par le service séparé
"pricer-service" (Jour 1/3, cf. pricer_service2.py), que ce point de terminaison appelle à distance
via modal.Cls.from_name — cet appel inter-apps est authentifié automatiquement par Modal, sans
configuration supplémentaire.

Une fois déployé, Modal affiche l'URL du point de terminaison, de la forme :
    https://<votre-espace>--agence-bon-prix-estimate.modal.run

Appel depuis n'importe quel outil (curl, Slack bot, extension navigateur, CRM...) :
    curl -X POST https://<votre-espace>--agence-bon-prix-estimate.modal.run \\
         -H "Content-Type: application/json" \\
         -d '{"description": "iPhone 12, 64GB, débloqué, bon état", "asking_price": 250}'

Sécurisation optionnelle (recommandée avant une mise en production réelle) : créez un secret
    modal secret create agence-api-key AGENCE_API_KEY=votre-cle-partagee
puis ajoutez-le à la liste `secrets=[...]` ci-dessous. Tant qu'il n'est pas configuré, l'API reste
ouverte (pas d'authentification) — pratique pour la démo du cours, à corriger avant tout usage réel.
"""

import os

import modal
from fastapi import Header, HTTPException
from pydantic import BaseModel

app = modal.App("agence-bon-prix")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "langgraph",
        "langchain",
        "langchain-openai",
        "openai",
        "chromadb",
        "sentence-transformers",
        "pydantic",
        "fastapi",
    )
    # products_vectorstore/ (base Chroma, ~22 Mo) : embarquée dans l'image pour que le RAG
    # fonctionne dans le conteneur, sans dépendre d'un disque partagé.
    .add_local_dir("products_vectorstore", remote_path="/root/products_vectorstore")
    # Le paquet agents/ (Jour 3) et notre module d'orchestration (LangGraph, Jour 2).
    .add_local_python_source("agents", "agence_workflow")
)

secrets = [modal.Secret.from_name("openai-secret")]


class EstimateRequest(BaseModel):
    description: str
    asking_price: float | None = None


@app.function(image=image, secrets=secrets, timeout=300)
@modal.fastapi_endpoint(method="POST", docs=True)
def estimate(request: EstimateRequest, x_api_key: str | None = Header(default=None)) -> dict:
    """POST /estimate — {"description": "...", "asking_price": 250.0}  (asking_price optionnel)."""
    expected_key = os.environ.get("AGENCE_API_KEY")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Clé API invalide ou manquante (en-tête X-API-Key).")

    if not request.description.strip():
        raise HTTPException(status_code=400, detail="Le champ 'description' est requis.")

    from agence_workflow import run_estimation

    return run_estimation(request.description, asking_price=request.asking_price)

import os
import sys
import logging
import json
from typing import List
from dotenv import load_dotenv
import chromadb
from agents.planning_agent import PlanningAgent
from agents.deals import Opportunity
from sklearn.manifold import TSNE
import numpy as np

load_dotenv(override=True)

# Couleurs pour les logs (codes ANSI)
BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

# Couleurs pour le graphique (une par catégorie de produit)
CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"]


def init_logging():
    # Configure la journalisation : niveau INFO, sortie vers la console, avec un format horodaté
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class DealAgentFramework:
    # Cadre qui orchestre les agents et conserve une "mémoire" des opportunités déjà trouvées.
    DB = "products_vectorstore"          # Dossier de la base vectorielle Chroma
    MEMORY_FILENAME = "memory.json"      # Fichier de mémoire persistante (opportunités passées)

    def __init__(self):
        init_logging()
        client = chromadb.PersistentClient(path=self.DB)
        self.memory = self.read_memory()                                  # Charge les opportunités mémorisées
        self.collection = client.get_or_create_collection("products")     # Collection de produits (RAG)
        self.planner = None

    def init_agents_as_needed(self):
        # Initialise l'agent planificateur seulement s'il ne l'est pas déjà (chargement coûteux)
        if not self.planner:
            self.log("Initialisation du cadre d'agents")
            self.planner = PlanningAgent(self.collection)
            self.log("Cadre d'agents prêt")

    def read_memory(self) -> List[Opportunity]:
        # Lit la mémoire (opportunités) depuis le fichier JSON, si présent
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
            opportunities = [Opportunity(**item) for item in data]
            return opportunities
        return []

    def write_memory(self) -> None:
        # Écrit la mémoire courante dans le fichier JSON
        data = [opportunity.model_dump() for opportunity in self.memory]
        with open(self.MEMORY_FILENAME, "w") as file:
            json.dump(data, file, indent=2)

    @classmethod
    def reset_memory(cls) -> None:
        # Réinitialise la mémoire en ne conservant que les 2 premières opportunités
        data = []
        if os.path.exists(cls.MEMORY_FILENAME):
            with open(cls.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
        truncated = data[:2]
        with open(cls.MEMORY_FILENAME, "w") as file:
            json.dump(truncated, file, indent=2)

    def log(self, message: str):
        # Journalise un message coloré, préfixé par [Agent Framework]
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def run(self) -> List[Opportunity]:
        # Lance un cycle complet : le planificateur cherche une opportunité, qu'on ajoute à la mémoire
        self.init_agents_as_needed()
        logging.info("Démarrage de l'agent planificateur")
        result = self.planner.plan(memory=self.memory)
        logging.info(f"L'agent planificateur a terminé et a renvoyé : {result}")
        if result:
            self.memory.append(result)
            self.write_memory()
        return self.memory

    @classmethod
    def get_plot_data(cls, max_datapoints=2000):
        # Prépare les données pour la visualisation 3D : récupère les vecteurs et les projette en 3D via t-SNE
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection("products")
        result = collection.get(
            include=["embeddings", "documents", "metadatas"], limit=max_datapoints
        )
        vectors = np.array(result["embeddings"])
        documents = result["documents"]
        categories = [metadata["category"] for metadata in result["metadatas"]]
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors


if __name__ == "__main__":
    DealAgentFramework().run()

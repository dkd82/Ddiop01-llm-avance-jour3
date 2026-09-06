import re
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from agents.agent import Agent


class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = Agent.BLUE

    MODEL = "gpt-4o-mini"

    def __init__(self, collection):
        """
        Initialise cette instance : connexion à OpenAI, à la base Chroma,
        et mise en place du modèle d'encodage vectoriel.
        """
        self.log("Initialisation du Frontier Agent")
        self.client = OpenAI()
        self.MODEL = "gpt-5.1"
        self.log("Le Frontier Agent se configure avec OpenAI")
        self.collection = collection
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.log("Le Frontier Agent est prêt")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Crée le contexte à insérer dans le prompt.
        :param similars: produits similaires à celui que l'on estime
        :param prices: prix de ces produits similaires
        :return: le texte de contexte à insérer dans le prompt
        """
        # NB : ce texte est en anglais car il est envoyé au LLM avec des descriptions de produits en anglais.
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(
        self, description: str, similars: List[str], prices: List[float]
    ) -> List[Dict[str, str]]:
        """
        Crée la liste de messages pour l'appel à OpenAI (prompt système + utilisateur).
        :param description: description du produit
        :param similars: produits similaires
        :param prices: prix des produits similaires
        :return: la liste de messages au format attendu par OpenAI
        """
        # Consigne en anglais (le produit et le contexte sont en anglais)
        message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{description}\n\n"
        message += self.make_context(similars, prices)
        return [{"role": "user", "content": message}]

    def find_similars(self, description: str):
        """
        Renvoie une liste de produits similaires à celui fourni, en interrogeant la base Chroma.
        """
        self.log(
            "Le Frontier Agent effectue une recherche RAG dans Chroma pour trouver 5 produits similaires"
        )
        vector = self.model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results["documents"][0][:]
        prices = [m["price"] for m in results["metadatas"][0][:]]
        self.log("Le Frontier Agent a trouvé des produits similaires")
        return documents, prices

    def get_price(self, s) -> float:
        """
        Utilitaire qui extrait un nombre à virgule flottante d'une chaîne de caractères.
        """
        s = s.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        """
        Appelle OpenAI pour estimer le prix du produit décrit, en recherchant 5 produits similaires
        et en les incluant dans le prompt afin de fournir du contexte (RAG).
        :param description: description du produit
        :return: une estimation du prix
        """
        documents, prices = self.find_similars(description)
        self.log(
            f"Le Frontier Agent s'apprête à appeler {self.MODEL} avec un contexte de 5 produits similaires"
        )
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self.messages_for(description, documents, prices),
            seed=42,
            reasoning_effort="none",
        )
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Frontier Agent terminé — prédiction de ${result:.2f}")
        return result

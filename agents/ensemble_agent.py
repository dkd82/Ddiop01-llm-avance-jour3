from agents.agent import Agent
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.neural_network_agent import NeuralNetworkAgent
from agents.preprocessor import Preprocessor


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection):
        """
        Crée une instance d'ensemble en créant chacun des modèles
        (spécialiste, frontier/RAG, réseau de neurones) et le préprocesseur.
        """
        self.log("Initialisation de l'agent d'ensemble")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.neural_network = NeuralNetworkAgent()
        self.preprocessor = Preprocessor()
        self.log("L'agent d'ensemble est prêt")

    def price(self, description: str) -> float:
        """
        Exécute le modèle d'ensemble : on demande à chaque modèle d'estimer le prix du produit,
        puis on renvoie une moyenne pondérée des estimations.
        :param description: la description d'un produit
        :return: une estimation de son prix
        """
        self.log("Exécution de l'agent d'ensemble — prétraitement du texte")
        rewrite = self.preprocessor.preprocess(description)   # Normalise la description
        self.log(f"Texte prétraité avec {self.preprocessor.model_name}")
        specialist = self.specialist.price(rewrite)
        frontier = self.frontier.price(rewrite)
        neural_network = self.neural_network.price(rewrite)
        # Moyenne pondérée : 80 % RAG (frontier), 10 % spécialiste, 10 % réseau de neurones
        combined = frontier * 0.8 + specialist * 0.1 + neural_network * 0.1
        self.log(f"Agent d'ensemble terminé — renvoie ${combined:.2f}")
        return combined

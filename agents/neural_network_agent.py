from agents.agent import Agent
from agents.deep_neural_network import DeepNeuralNetworkInference


class NeuralNetworkAgent(Agent):
    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self):
        """
        Initialise cet objet en chargeant les poids enregistrés du modèle
        et le modèle d'encodage vectoriel SentenceTransformer.
        """
        self.log("L'agent réseau de neurones s'initialise")
        self.neural_network = DeepNeuralNetworkInference()
        self.neural_network.setup()
        self.neural_network.load("deep_neural_network.pth")
        self.log("L'agent réseau de neurones est prêt et les poids sont chargés")

    def price(self, description: str) -> float:
        """
        Utilise le réseau de neurones profond pour estimer le prix de l'article décrit.
        :param description: le produit à estimer
        :return: le prix sous forme de nombre flottant
        """
        self.log("L'agent réseau de neurones démarre une prédiction")
        result = self.neural_network.inference(description)
        self.log(f"Agent réseau de neurones terminé — prédiction de ${result:.2f}")
        return result

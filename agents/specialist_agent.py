import modal
from agents.agent import Agent


class SpecialistAgent(Agent):
    """
    Agent qui exécute notre LLM fine-tuné, hébergé à distance sur Modal.
    """

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        """
        Initialise cet agent en créant une instance de la classe Modal.
        """
        self.log("L'agent spécialiste s'initialise — connexion à Modal")
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()

    def price(self, description: str) -> float:
        """
        Effectue un appel distant pour renvoyer l'estimation de prix de cet article.
        """
        self.log("L'agent spécialiste appelle le modèle fine-tuné distant")
        result = self.pricer.price.remote(description)
        self.log(f"Agent spécialiste terminé — prédiction de ${result:.2f}")
        return result

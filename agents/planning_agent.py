from typing import Optional, List
from agents.agent import Agent
from agents.deals import ScrapedDeal, DealSelection, Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent


class PlanningAgent(Agent):

    name = "Planning Agent"
    color = Agent.GREEN
    DEAL_THRESHOLD = 50   # Remise minimale (en $) pour qu'une offre déclenche une notification

    def __init__(self, collection):
        """
        Crée les instances des 3 agents que ce planificateur coordonne.
        """
        self.log("L'agent planificateur s'initialise")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.log("L'agent planificateur est prêt")

    def run(self, deal: Deal) -> Opportunity:
        """
        Exécute le workflow pour une offre donnée.
        :param deal: l'offre, résumée à partir d'un flux RSS
        :returns: une opportunité incluant la remise estimée
        """
        self.log("L'agent planificateur évalue le prix d'une offre potentielle")
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price   # Remise = valeur estimée - prix proposé
        self.log(f"L'agent planificateur a traité une offre avec une remise de ${discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        """
        Exécute le workflow complet :
        1. Utiliser le ScannerAgent pour trouver des offres dans les flux RSS
        2. Utiliser l'EnsembleAgent pour les estimer
        3. Utiliser le MessagingAgent pour envoyer une notification
        :param memory: liste d'URL déjà signalées par le passé
        :return: une opportunité si l'une a été retenue, sinon None
        """
        self.log("L'agent planificateur démarre un cycle")
        selection = self.scanner.scan(memory=memory)
        if selection:
            # On estime les 5 meilleures offres, puis on les trie par remise décroissante
            opportunities = [self.run(deal) for deal in selection.deals[:5]]
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)
            best = opportunities[0]
            self.log(f"L'agent planificateur a identifié la meilleure offre, remise de ${best.discount:.2f}")
            # On ne notifie que si la remise dépasse le seuil
            if best.discount > self.DEAL_THRESHOLD:
                self.messenger.alert(best)
            self.log("L'agent planificateur a terminé un cycle")
            return best if best.discount > self.DEAL_THRESHOLD else None
        return None

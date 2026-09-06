import os
from agents.deals import Opportunity
from agents.agent import Agent
import requests

pushover_url = "https://api.pushover.net/1/messages.json"


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.WHITE
    MODEL = "claude-sonnet-4-5"

    def __init__(self):
        """
        Initialise cet objet pour envoyer soit des notifications push via Pushover,
        soit des SMS via Twilio, selon ce qui est configuré dans les constantes.
        """
        self.log("L'agent de messagerie s'initialise")
        self.pushover_user = os.getenv("PUSHOVER_USER", "your-pushover-user-if-not-using-env")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN", "your-pushover-user-if-not-using-env")
        self.log("L'agent de messagerie a initialisé Pushover et Claude")

    def push(self, text):
        """
        Envoie une notification push via l'API Pushover.
        """
        self.log("L'agent de messagerie envoie une notification push")
        payload = {
            "user": self.pushover_user,
            "token": self.pushover_token,
            "message": text,
            "sound": "cashregister",   # Son de "tiroir-caisse" pour les bonnes affaires
        }
        requests.post(pushover_url, data=payload)

    def alert(self, opportunity: Opportunity):
        """
        Émet une alerte au sujet de l'opportunité spécifiée.
        """
        text = f"Alerte bonne affaire ! Prix=${opportunity.deal.price:.2f}, "
        text += f"Estimation=${opportunity.estimate:.2f}, "
        text += f"Remise=${opportunity.discount:.2f} :"
        text += opportunity.deal.product_description[:10] + "... "
        text += opportunity.deal.url
        self.push(text)
        self.log("L'agent de messagerie a terminé")

    def craft_message(
        self, description: str, deal_price: float, estimated_true_value: float
    ) -> str:
        # Prompt rédigé en français : la notification finale sera donc en français.
        user_prompt = "Résume cette excellente offre en 2 ou 3 phrases, sous la forme d'une notification push enthousiasmante pour alerter l'utilisateur.\n"
        user_prompt += f"Description de l'article : {description}\nPrix proposé : {deal_price}\nValeur réelle estimée : {estimated_true_value}"
        user_prompt += "\n\nRéponds uniquement avec le message de 2-3 phrases qui servira à alerter et enthousiasmer l'utilisateur au sujet de cette offre."
        try:
            from litellm import completion

            response = completion(
                model=self.MODEL,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception:
            savings = max(estimated_true_value - deal_price, 0)
            return (
                f"Bonne affaire repérée : {description[:80]}... proposé à ${deal_price:.2f}. "
                f"Valeur estimée ${estimated_true_value:.2f}, soit environ ${savings:.2f} d'économie potentielle."
            )

    def notify(self, description: str, deal_price: float, estimated_true_value: float, url: str):
        """
        Émet une alerte à partir des détails spécifiés.
        """
        self.log("L'agent de messagerie utilise Claude pour rédiger le message")
        text = self.craft_message(description, deal_price, estimated_true_value)
        self.push(text[:200] + "... " + url)
        self.log("L'agent de messagerie a terminé")

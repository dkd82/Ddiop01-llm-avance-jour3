import logging

class Agent:
    """
    Super-classe abstraite pour les agents.
    Sert à journaliser les messages d'une manière qui permet d'identifier chaque agent.
    """

    # Couleurs de premier plan (codes ANSI)
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Couleur d'arrière-plan
    BG_BLACK = '\033[40m'

    # Code de réinitialisation (retour à la couleur par défaut)
    RESET = '\033[0m'

    name: str = ""
    color: str = '\033[37m'

    def log(self, message):
        """
        Journalise ce message en niveau INFO, en identifiant l'agent (par son nom et sa couleur).
        """
        color_code = self.BG_BLACK + self.color
        message = f"[{self.name}] {message}"
        logging.info(color_code + message + self.RESET)

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
BG_BLUE = '\033[44m'

# Code de réinitialisation (retour à la couleur par défaut)
RESET = '\033[0m'

# Table de correspondance : combinaison de codes ANSI -> couleur HTML (pour l'affichage dans l'interface web)
mapper = {
    BG_BLACK+RED: "#dd0000",
    BG_BLACK+GREEN: "#00dd00",
    BG_BLACK+YELLOW: "#dddd00",
    BG_BLACK+BLUE: "#0000ee",
    BG_BLACK+MAGENTA: "#aa00dd",
    BG_BLACK+CYAN: "#00dddd",
    BG_BLACK+WHITE: "#87CEEB",
    BG_BLUE+WHITE: "#ff7800"
}


def reformat(message):
    # Convertit les codes couleur ANSI d'un message de log en balises HTML <span> colorées
    for key, value in mapper.items():
        message = message.replace(key, f'<span style="color: {value}">')
    message = message.replace(RESET, '</span>')
    return message

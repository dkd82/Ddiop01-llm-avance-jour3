import modal
from modal import Image

# Configuration

app = modal.App("hello")
image = Image.debian_slim().pip_install("requests")  # Image Debian minimale + la bibliothèque requests

# Hello !


@app.function(image=image)
def hello() -> str:
    # Cette fonction s'exécute à distance sur Modal : elle interroge un service web
    # pour connaître la ville/région/pays du serveur, puis renvoie un message.
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    city, region, country = data["city"], data["region"], data["country"]
    return f"Bonjour depuis {city}, {region}, {country} !!"


# Nouveau - ajouté grâce à l'étudiant Tue H. !


@app.function(image=image, region="eu")
def hello_europe() -> str:
    # Même fonction, mais exécutée sur un serveur situé en Europe (region="eu")
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    city, region, country = data["city"], data["region"], data["country"]
    return f"Bonjour depuis {city}, {region}, {country} !!"

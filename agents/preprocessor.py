import os
import re

from dotenv import load_dotenv

load_dotenv(override=True)

# Modèle utilisé par défaut pour le prétraitement (surchargeable via la variable d'env PRICER_PREPROCESSOR_MODEL)
DEFAULT_MODEL_NAME = os.getenv("PRICER_PREPROCESSOR_MODEL", "ollama/llama3.2")
DEFAULT_REASONING_EFFORT = "low" if "gpt-oss" in DEFAULT_MODEL_NAME else None

# ⚠️ Prompt conservé en anglais : il produit une description structurée (Title/Category/...) qui alimente
# les modèles d'estimation entraînés sur des données anglaises. Le traduire dégraderait les prédictions.
SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Preprocessor:
    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        base_url=None,
    ):
        # Compteurs de tokens et de coût, pour le suivi de la consommation
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        # Si on utilise Ollama en local sans URL fournie, on pointe vers le serveur Ollama par défaut
        if "ollama" in model_name and not base_url:
            self.base_url = "http://localhost:11434"

    def messages_for(self, text: str) -> list[dict]:
        # Construit la liste de messages : la consigne système + le texte à reformuler
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]

    @staticmethod
    def _guess_category(text: str) -> str:
        lowered = text.lower()
        category_keywords = {
            "Electronics": [
                "iphone", "phone", "laptop", "usb", "keyboard", "mouse", "monitor", "mic",
                "headphone", "speaker", "camera", "tv", "tablet", "router",
            ],
            "Musical Instruments": ["guitar", "piano", "mic", "microphone", "drum", "midi"],
            "Office Products": ["printer", "paper", "office", "desk", "chair", "stapler"],
            "Tools and Home Improvement": ["drill", "saw", "tool", "wrench", "hammer"],
            "Appliances": ["fridge", "microwave", "blender", "vacuum", "toaster", "oven"],
            "Toys and Games": ["lego", "toy", "game", "puzzle", "controller", "console"],
            "Automotive": ["car", "vehicle", "dashcam", "tire", "engine", "automotive"],
        }
        for category, keywords in category_keywords.items():
            if any(keyword in lowered for keyword in keywords):
                return category
        return "Electronics"

    @staticmethod
    def _guess_brand(text: str) -> str:
        known_brands = [
            "Apple", "Samsung", "Sony", "Bose", "HyperX", "Logitech", "Dell", "HP",
            "Lenovo", "Asus", "Acer", "Canon", "Nikon", "Microsoft", "Google", "Anker",
        ]
        for brand in known_brands:
            if brand.lower() in text.lower():
                return brand

        title_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+\-.]*", text)
        for word in title_words:
            if any(char.isalpha() for char in word) and word[:1].isupper():
                return word
        return "Unknown"

    def _fallback_preprocess(self, text: str) -> str:
        compact = " ".join(text.split())
        words = compact.split()
        title = " ".join(words[:8]).strip(" ,.-") or "Product"
        category = self._guess_category(compact)
        brand = self._guess_brand(compact)
        description = compact[:180].rstrip(" ,.-")
        details = compact[:220].rstrip(" ,.-")
        return (
            f"Title: {title}\n"
            f"Category: {category}\n"
            f"Brand: {brand}\n"
            f"Description: {description}\n"
            f"Details: {details}"
        )

    def preprocess(self, text: str) -> str:
        # Appelle le modèle pour produire une description normalisée, et met à jour les compteurs
        messages = self.messages_for(text)
        try:
            from litellm import completion

            response = completion(
                messages=messages,
                model=self.model_name,
                reasoning_effort=self.reasoning_effort,
                api_base=self.base_url,
            )
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            self.total_cost += response._hidden_params.get("response_cost", 0)
            return response.choices[0].message.content
        except Exception as e:
            print(
                f"⚠️ Échec de l'appel au modèle de prétraitement '{self.model_name}'"
                + (f" (api_base={self.base_url})" if self.base_url else "")
                + f" : {e}\n"
                "   → Bascule sur un prétraitement de secours (heuristique, qualité dégradée) : "
                "la description générée sera de moins bonne qualité et l'estimation de prix qui en "
                "découle sera probablement peu fiable.\n"
                "   → Si vous comptiez utiliser Ollama, vérifiez qu'il est bien démarré en local "
                "('ollama serve', puis 'ollama pull llama3.2').\n"
                "   → Alternative sans installation locale : "
                "Preprocessor(model_name='groq/openai/gpt-oss-20b')."
            )
            return self._fallback_preprocess(text)

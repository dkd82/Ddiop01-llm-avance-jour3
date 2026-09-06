import modal
from modal import Volume, Image
# Configuration - on définit notre infrastructure directement en code !

app = modal.App("pricer-service")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)

# Récupère le secret depuis Modal.
# Selon votre configuration Modal, vous devrez peut-être remplacer "huggingface-secret" par "hf-secret".
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "Ddiop01"  # Votre nom Hugging Face ici ! Ou gardez le mien pour reproduire mes résultats.
RUN_NAME = "2026-06-09_10.08.11-lite"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "48bdfbf9c81faf1e48b55a0d0044efb297745a1a"  # Révision du modèle fine-tuné sur Hugging Face (à mettre à jour si vous entraînez votre propre modèle). 
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"
CACHE_DIR = "/cache"

# Mettre à 1 si vous voulez que Modal reste toujours actif ; sinon le conteneur "refroidit" après 2 min.
MIN_CONTAINERS = 0

# ⚠️ Prompt EXACT d'entraînement du modèle (en anglais) — à NE PAS traduire.
PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"

# Volume persistant pour mettre en cache les poids Hugging Face (évite de les retélécharger à chaque démarrage)
hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Pricer:
    @modal.enter()
    def setup(self):
        # @modal.enter() : exécuté UNE fois au démarrage du conteneur -> on y charge le modèle (coûteux),
        # pour ne pas le recharger à chaque appel de price().
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        # Configuration de quantification 4 bits
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        # Chargement du modèle et du tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quant_config, device_map="auto"
        )
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model, FINETUNED_MODEL, revision=REVISION
        )

    @modal.method()
    def price(self, description: str) -> float:
        # Estime le prix : le modèle étant déjà chargé (setup), cet appel est rapide.
        import re
        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")
        with torch.no_grad():
            outputs = self.fine_tuned_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        result = self.tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False)
        contents = result.split("Price is $")[1]
        contents = contents.replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0

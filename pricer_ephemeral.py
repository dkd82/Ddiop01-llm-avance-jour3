import modal
from modal import Image

# Configuration

app = modal.App("pricer")
# Image avec les bibliothèques nécessaires à la quantification (bitsandbytes) et aux adaptateurs LoRA (peft)
image = Image.debian_slim().pip_install(
    "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)
secrets = [modal.Secret.from_name("huggingface-secret")]

# Constantes

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "Ddiop01"  # Votre nom Hugging Face ici ! Ou gardez le mien pour reproduire mes résultats.
RUN_NAME = "2026-06-23_20.55.07"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = None   # None = dernière révision du dépôt (pas de commit précis connu pour ce run)
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"


@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def price(description: str) -> float:
    # Estime le prix d'un produit décrit par `description`, à l'aide du modèle fine-tuné en semaine 7.
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel

    # ⚠️ Ces deux chaînes constituent le PROMPT EXACT sur lequel le modèle a été entraîné (en anglais).
    # Il ne faut PAS les traduire, sous peine de dégrader fortement les prédictions.
    PREFIX = "Price is $"
    QUESTION = "What does this cost to the nearest dollar?"

    prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

    # Configuration de quantification 4 bits (QLoRA)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    # Chargement du modèle et du tokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config, device_map="auto"
    )

    # On greffe l'adaptateur LoRA fine-tuné par-dessus le modèle de base
    fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL, revision=REVISION)

    set_seed(42)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    with torch.no_grad():
        outputs = fine_tuned_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            pad_token_id=tokenizer.pad_token_id,
        )
    result = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False)
    # On récupère le texte généré après "Price is $", puis on en extrait le nombre
    contents = result.split("Price is $")[1]
    contents = contents.replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0

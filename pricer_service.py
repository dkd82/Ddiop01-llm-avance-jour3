import modal
from modal import Image

# Configuration - on définit notre infrastructure directement en code !

app = modal.App("pricer-service")
image = Image.debian_slim().pip_install(
    "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)

# Récupère le secret depuis Modal.
# Selon votre configuration Modal, vous devrez peut-être remplacer "huggingface-secret" par "hf-secret".
secrets = [modal.Secret.from_name("huggingface-secret")]

# Constantes

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "Ddiop01"  # Votre nom Hugging Face ici ! Ou gardez le mien pour reproduire mes résultats.
RUN_NAME = "2026-06-23_20.55.07"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = None  # None = dernière révision du dépôt (pas de commit précis connu pour ce run)
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"


@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def price(description: str) -> float:
    # Version "déployée" du service de tarification (même logique que pricer_ephemeral).
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel

    # ⚠️ Prompt EXACT d'entraînement du modèle (en anglais) — à NE PAS traduire.
    PREFIX = "Price is $"
    QUESTION = "What does this cost to the nearest dollar?"

    prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

    # Configuration de quantification 4 bits
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
    contents = result.split("Price is $")[1]
    contents = contents.replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0

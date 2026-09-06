import modal
from modal import Image

# Configuration

app = modal.App("llama")
# Image avec PyTorch, Transformers et Accelerate (nécessaires pour charger et exécuter le modèle)
image = Image.debian_slim().pip_install("torch", "transformers", "accelerate")
secrets = [modal.Secret.from_name("huggingface-secret")]  # Jeton Hugging Face (secret Modal)
GPU = "T4"                                                  # Type de GPU demandé
MODEL_NAME = "meta-llama/Llama-3.2-3B"


@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def generate(prompt: str) -> str:
    # Fonction exécutée à distance sur un GPU Modal : charge Llama et complète le prompt.
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token   # Pas de token de padding dédié -> on réutilise le token de fin
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    set_seed(42)                                 # Graine fixe pour des résultats reproductibles
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        pad_token_id=tokenizer.pad_token_id,
    )  # On ne génère que quelques tokens
    return tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False)

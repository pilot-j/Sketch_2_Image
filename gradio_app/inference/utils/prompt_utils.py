import torch
from transformers import CLIPTokenizer, CLIPTextModel

def encode_prompt(prompt: str, tokenizer, text_encoder, device):
    inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        return text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state

def build_prompt(color: str = "", structure: str = "", material: str = "") -> str:
    structure = structure if structure.strip() else "building"
    material = material if material.strip() else "concrete"

    if color.strip():
        prompt = f"{color} {structure} made of {material}"
    else:
        prompt = f"{structure} made of {material}"

    return prompt

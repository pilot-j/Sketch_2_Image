from huggingface_hub import hf_hub_download
import gradio as gr
from PIL import Image
import torch
from inference.utils.inference_utils import run_controlnet_inference
from inference.utils.prompt_utils import build_prompt
from inference.lora.lora_layers import init_lora_attn
from diffusers import DDIMScheduler, UNet2DConditionModel, AutoencoderKL, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
import os

# Paths
pretrained_model_path = "runwayml/stable-diffusion-v1-5"
controlnet_model_path = "lllyasviel/sd-controlnet-canny"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float32

# Load models once
def load_models():
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet").to(device)
    controlnet = ControlNetModel.from_pretrained(controlnet_model_path).to(device)
    init_lora_attn(controlnet, lora_rank = 512)  # Setup LoRA-compatible layers

    return {
        "noise_scheduler": noise_scheduler,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "controlnet": controlnet,
        "device": device,
        "weight_dtype": weight_dtype,
    }

models = load_models()

# Wrapper for inference
def infer(image, color, structure, material):
    if image is None:
        return None

    repo_id = "pilotj/control_net"
    filename = "raw_wts_v1_controlnet"
    wts_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    structure = structure.strip() if structure.strip() else "building"
    color = color.strip() if color.strip() else "white"
    material = material.strip() if material.strip() else "concrete"

    prompt = f"{structure} of {color} color made of {material}"

    return run_controlnet_inference(
        prompt=prompt,
        control_image=image,
        noise_scheduler=models["noise_scheduler"],
        tokenizer=models["tokenizer"],
        text_encoder=models["text_encoder"],
        vae=models["vae"],
        unet=models["unet"],
        controlnet=models["controlnet"],
        wts_path=wts_path,
        device=models["device"],
        weight_dtype=models["weight_dtype"],
        inject_lora=False
    )

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Sketch to Building Image Generator")
    with gr.Row():
        image_input = gr.Image(type="pil", label="Sketch Input")
        image_output = gr.Image(label="Generated Image")

    with gr.Row():
        color = gr.Textbox(label="Color (optional)")
        structure = gr.Textbox(label="Structure (optional)")
        material = gr.Textbox(label="Material (optional)")

    generate_btn = gr.Button("Generate")

    generate_btn.click(
        fn=infer,
        inputs=[image_input, color, structure, material],
        outputs=image_output
    )

demo.launch()

import torch
from diffusers import DDIMScheduler, UNet2DConditionModel, AutoencoderKL, ControlNetModel
from transformers import CLIPTokenizer, CLIPTextModel

def load_models(pretrained_model_path="runwayml/stable-diffusion-v1-5", 
                controlnet_model_path="lllyasviel/sd-controlnet-canny"):
    
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(controlnet_model_path)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32

    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)
    controlnet.eval()

    return {
        "device": device,
        "weight_dtype": weight_dtype,
        "noise_scheduler": noise_scheduler,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "controlnet": controlnet
    }

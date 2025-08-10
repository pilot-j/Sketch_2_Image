import warnings
import torch
from diffusers import DDIMScheduler, UNet2DConditionModel, AutoencoderKL, ControlNetModel
from transformers import CLIPTokenizer, CLIPTextModel
from utils import run_controlnet_inference, show_images_side_by_side
from lora import init_lora_attn, inject_lora_adapter, setup_lora

warnings.filterwarnings("ignore")
 


def init_models(pretrained_model_path, controlnet_model_path, device, dtype):
    # Load scheduler & tokenizer
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")

    # Load text encoder, VAE, UNet, ControlNet
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(controlnet_model_path)

    # Freeze non-LoRA parameters
    for m in [vae, text_encoder, unet]:
        m.requires_grad_(False)

    # Move to device
    for m in [text_encoder, vae, unet]:
        m.to(device, dtype=dtype)

    return noise_scheduler, tokenizer, text_encoder, vae, unet, controlnet


## Single Image Generation
def generate_single_image(
    prompt, control_image, noise_scheduler, tokenizer, text_encoder, vae, unet, controlnet,
    device, dtype, steps=50
):
    controlnet.eval()
    with torch.no_grad():
        return run_controlnet_inference(
            prompt=prompt,
            control_image=control_image,
            noise_scheduler=noise_scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            num_inference_steps=steps,
            device=device,
            weight_dtype=dtype
        )

if __name__ == "__main__":
    pretrained_model_path = "runwayml/stable-diffusion-v1-5"
    controlnet_model_path = "lllyasviel/sd-controlnet-canny"
    lora_path = "/kaggle/input/lora-weights-full/lora_adapter_v2.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32

    # Init models
    noise_scheduler, tokenizer, text_encoder, vae, unet, controlnet = init_models(
        pretrained_model_path, controlnet_model_path, device, weight_dtype
    )

    # Inject LoRA
    setup_lora(controlnet, lora_path, lora_rank=512, device=device, dtype=weight_dtype)

    # Prompt & image
    prompt = "Rectangular house with flat roof made of concrete"
    val_img = Image.open("/kaggle/input/val-image/whereness assignment sketch.png")  # Load your control image here, replace file path.

    # Generate
    generated_image = generate_single_image(
        prompt, val_img, noise_scheduler, tokenizer, text_encoder, vae, unet, controlnet,
        device, weight_dtype, steps=70
    )

    # Show
    print(f"Guidance_prompt: {prompt}")
    show_images_side_by_side([val_img, generated_image], titles=["Sketch", "Generation"])

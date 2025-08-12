import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def preprocess_controlnet_image(image: Image.Image, device, dtype=torch.float32):
    """
    Resize and normalize an image for ControlNet input.
    Converts to tensor with shape (1, C, H, W).
    """
    image = image.resize((512, 512))
    img_array = np.array(image).astype(np.float32) / 255.0  # normalize to [0,1]
    
    if img_array.ndim == 2:  # Grayscale
        img_array = img_array[None, None, :, :]
    elif img_array.shape[2] == 3:  # RGB
        img_array = img_array.transpose(2, 0, 1)[None, :, :, :]
    else:
        raise ValueError(f"Unexpected image shape: {img_array.shape}")

    return torch.tensor(img_array, device=device, dtype=dtype)


def encode_prompt(prompt: str, tokenizer, text_encoder, device):
    """
    Tokenize and encode a text prompt into embeddings.
    """
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


@torch.no_grad()
def run_controlnet_inference(
    prompt: str,
    control_image: Image.Image,
    noise_scheduler,
    tokenizer,
    text_encoder,
    vae,
    unet,
    controlnet,
    num_inference_steps=50,
    guidance_scale=7.5,
    device=None,
    weight_dtype=torch.float32
):
    """
    Run a full ControlNet inference pipeline for a single prompt and control image.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Encode prompt and unconditional prompt
    cond_embeds = encode_prompt(prompt, tokenizer, text_encoder, device)
    uncond_embeds = encode_prompt("", tokenizer, text_encoder, device)
    prompt_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)

    # Preprocess controlnet image
    controlnet_image = preprocess_controlnet_image(control_image, device=device, dtype=weight_dtype)
    controlnet_image = torch.cat([controlnet_image, controlnet_image], dim=0)  # guidance duplication

    # Prepare latents
    batch_size = 1
    latent_shape = (batch_size, unet.in_channels, 64, 64)
    latents = torch.randn(latent_shape, device=device, dtype=weight_dtype) * noise_scheduler.init_noise_sigma

    noise_scheduler.set_timesteps(num_inference_steps)

    for t in noise_scheduler.timesteps:
        latent_input = torch.cat([latents] * 2)
        latent_input = noise_scheduler.scale_model_input(latent_input, t)

        # ControlNet forward
        down_block_res_samples, mid_block_res_sample = controlnet(
            latent_input,
            t,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=controlnet_image,
            return_dict=False,
        )

        # UNet forward
        noise_pred = unet(
            latent_input,
            t,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=[res.to(dtype=weight_dtype) for res in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
            return_dict=False,
        )[0]

        # Classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Latent update
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample

    # Postprocess
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


def show_images_side_by_side(images, titles=None):
    """
    Display a list of images side-by-side with optional titles.
    """
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axs = [axs]
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis('off')
        if titles is not None:
            axs[i].set_title(titles[i])
    plt.show()

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_optimizer(cfg, lora_layers):
    opt_type = cfg["optimizer"]["type"]
    opt_params = cfg["optimizer"]["params"]

    if not hasattr(torch.optim, opt_type):
        raise ValueError(f"Unknown optimizer type: {opt_type}")
    opt_class = getattr(torch.optim, opt_type)

    return opt_class(
        [p for layer in lora_layers for p in layer.parameters() if p.requires_grad],
        **opt_params
    )

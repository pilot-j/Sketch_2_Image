import torch
import numpy as np
from PIL import Image

from inference.utils.prompt_utils import encode_prompt
from inference.utils.image_utils import preprocess_controlnet_image

from inference.lora.lora_layers import init_lora_attn


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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")


    cond_embeds = encode_prompt(prompt, tokenizer, text_encoder, device)
    uncond_embeds = encode_prompt("", tokenizer, text_encoder, device)
    prompt_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)

    controlnet_image = preprocess_controlnet_image(control_image, device=device, dtype=weight_dtype)
    controlnet_image = torch.cat([controlnet_image, controlnet_image], dim=0)

    batch_size = 1
    latent_shape = (batch_size, unet.in_channels, 64, 64)
    latents = torch.randn(latent_shape, device=device, dtype=weight_dtype) * noise_scheduler.init_noise_sigma

    noise_scheduler.set_timesteps(num_inference_steps)

    for t in noise_scheduler.timesteps:
        latent_input = torch.cat([latents] * 2)
        latent_input = noise_scheduler.scale_model_input(latent_input, t)

        down_block_res_samples, mid_block_res_sample = controlnet(
            latent_input, t,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=controlnet_image,
            return_dict=False
        )

        noise_pred = unet(
            latent_input, t,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=[res.to(dtype=weight_dtype) for res in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
            return_dict=False,
        )[0]

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)

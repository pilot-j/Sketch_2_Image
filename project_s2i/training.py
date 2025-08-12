#!/usr/bin/env python3
"""
Training script for ControlNet with Modified LoRA.
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import random_split
from torchvision import transforms
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel
)
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from dataset import LoRADataset
from lora import  init_lora_attn, get_input
from utils import run_controlnet_inference, show_image, show_images_side_by_side

#Config
pretrained_model_path = "runwayml/stable-diffusion-v1-5"
controlnet_model_path = "lllyasviel/sd-controlnet-canny"
data_root = "/kaggle/input/sketch-2-image-dataset/construction_sketch_dataset/construction_sketch_dataset"
train_batch_size = 4
learning_rate = 1e-4
adam_beta1, adam_beta2 = 0.9, 0.999
adam_weight_decay = 1e-2
adam_epsilon = 1e-8
lora_rank = 512
gradient_accumulation_steps = 2
max_train_steps = 10000
lr_warmup_steps = 500
max_grad_norm = 1.0
fixed_image = "/kaggle/input/val-image/whereness assignment sketch.png"
seed = 42
save_loss_threshold = 0.09888
torch.manual_seed(seed)
np.random.seed(seed)

import torch
import math
import yaml
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import random_split
from torchvision import transforms
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler

from dataset import LoRADataset
from lora import init_lora_attn, get_input
from utils import run_controlnet_inference, show_image, show_images_side_by_side, load_config, create_optimizer


def main():
    cfg = load_config()

    # Reproducibility
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Load models
    noise_scheduler = DDIMScheduler.from_pretrained(cfg["pretrained_model_path"], subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(cfg["pretrained_model_path"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg["pretrained_model_path"], subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg["pretrained_model_path"], subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cfg["pretrained_model_path"], subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(cfg["controlnet_model_path"])

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    # LoRA
    lora_layers = init_lora_attn(controlnet, lora_rank=cfg["lora_rank"])

    # Optimizer from YAML
    optimizer = create_optimizer(cfg, lora_layers)

    # Load Data
    transform = transforms.Compose([transforms.Resize((512, 512))])
    full_dataset = LoRADataset(root=cfg["data_root"], split="", transform=transform)

    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["train_batch_size"], shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg["lr_warmup_steps"] * cfg["gradient_accumulation_steps"],
        num_training_steps=cfg["max_train_steps"] * cfg["gradient_accumulation_steps"],
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg["gradient_accumulation_steps"])
    num_train_epochs = math.ceil(cfg["max_train_steps"] / num_update_steps_per_epoch)

    controlnet.to(device)
    controlnet.train()

    global_step = 0
    best_loss = float("inf")

    # Training Loop
    for epoch in range(num_train_epochs):
        epoch_loss = 0.0
        steps_in_epoch = len(train_dataloader) // cfg["gradient_accumulation_steps"]
        progress_bar = tqdm(range(steps_in_epoch), desc=f"Epoch {epoch+1}/{num_train_epochs}")

        for step, batch in enumerate(train_dataloader):
            with torch.set_grad_enabled(True):
                x, controlnet_image, prompt_embed = get_input(batch)
                t = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (x.shape[0],),
                    device=device
                ).long()

                noise = torch.randn_like(x)
                x_noisy = noise_scheduler.add_noise(x, noise, t)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(x, noise, t)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                down_block_res_samples, mid_block_res_sample = controlnet(
                    x_noisy,
                    t,
                    encoder_hidden_states=prompt_embed,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                model_pred = unet(
                    x_noisy,
                    t,
                    encoder_hidden_states=prompt_embed,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss / cfg["gradient_accumulation_steps"]
                loss.backward()
                epoch_loss += loss.item()

                if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(lora_layers.parameters(), cfg["max_grad_norm"])
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    progress_bar.update(1)
                    global_step += 1
                    progress_bar.set_postfix(step_loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

            if global_step >= cfg["max_train_steps"]:
                break

        avg_epoch_loss = epoch_loss / steps_in_epoch
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss <= cfg["save_loss_threshold"]:
            save_path = f"./best_controlnet_{avg_epoch_loss:.5f}.pth"
            torch.save(controlnet.state_dict(), save_path)
            print(f"Saved ControlNet weights to {save_path}")
            best_loss = avg_epoch_loss

        if (epoch + 1) % 10 == 0:
            controlnet.eval()
            with torch.no_grad():
                generated_image = run_controlnet_inference(
                    prompt="Rectangular Structure House with flat roof",
                    control_image=cfg["fixed_image"],
                    noise_scheduler=noise_scheduler,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    controlnet=controlnet,
                    device=device,
                    weight_dtype=weight_dtype,
                    inject_lora=False
                )
                show_image(generated_image, title=f"Inference at Epoch {epoch+1}")
                show_images_side_by_side(
                    [cfg["fixed_image"], generated_image],
                    titles=["Fixed Control Image", f"Generation after Epoch {epoch+1}"]
                )
            controlnet.train()

        progress_bar.close()

if __name__ == "__main__":
    main()

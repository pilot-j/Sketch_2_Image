import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio
import torchvision.transforms as transforms  # if using torchvision transforms


class LoRADataset(Dataset):
    def __init__(self, root, prompt_json_path='/kaggle/input/sketch-2-image-dataset/prompts.json', default_prompt="", split="", length=None, transform=None):
        self.dir = os.path.join(root, split)
        self.img_names = os.listdir(os.path.join(self.dir, "Images"))
        self.default_prompt = default_prompt
        self.transform = transform

        # Load prompts from JSON
        self.prompt_dict = {}
        if prompt_json_path and os.path.exists(prompt_json_path):
            with open(prompt_json_path, "r") as f:
                prompts = json.load(f)
                self.prompt_dict = {item["image"]: item["prompt"] for item in prompts}

        np.random.shuffle(self.img_names)
        if length:
            self.img_names = self.img_names[:length]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        # Load target image
        target_path = os.path.join(self.dir, "Images", img_name)
        target_img = imageio.imread(target_path)[:,:,:3] #images are in png which usually have 4 channels - RGBA. We need only RGB.
        target_img = target_img.astype(np.float32) / 127.5 - 1.0
        target_img = torch.from_numpy(target_img).permute(2, 0, 1)

        # Load sketch image
        sketch_name = img_name.replace("image", "sketch")
        sketch_path = os.path.join(self.dir, "Sketches", sketch_name)
        if not os.path.exists(sketch_path):
            raise FileNotFoundError(f"Sketch not found for image {img_name}: {sketch_path}")

        sketch_img = imageio.imread(sketch_path)
        sketch_img = sketch_img.astype(np.float32) / 255.0
        sketch_img = torch.from_numpy(sketch_img).permute(2, 0, 1)

        # Get prompt from JSON or fallback to default
        prompt = self.prompt_dict.get(img_name, self.default_prompt)
        if not prompt:
            prompt = self.default_prompt  # fallback if prompt is empty string

        if self.transform:
            target_img = self.transform(target_img)
            sketch_img = self.transform(sketch_img)

        return {
            "image": target_img,
            "hint": sketch_img,
            "prompt": prompt
        }

from PIL import Image
import numpy as np
import torch

def preprocess_controlnet_image(image: Image.Image, device, dtype=torch.float32):
    image = image.resize((512, 512))
    img_array = np.array(image).astype(np.float32) / 255.0

    if img_array.ndim == 2:
        img_array = img_array[None, None, :, :]
    elif img_array.shape[2] == 3:
        img_array = img_array.transpose(2, 0, 1)[None, :, :, :]
    else:
        raise ValueError("Unexpected image shape.")

    return torch.tensor(img_array, device=device, dtype=dtype)

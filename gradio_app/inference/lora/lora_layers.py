import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank=256, alpha=1.0):
        super().__init__()
        if not isinstance(original_linear, nn.Linear):
            raise TypeError(f"LoRALinear only supports nn.Linear, but got: {type(original_linear)}")

        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, rank) * 0.01)

        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)


def init_lora_attn(model, lora_rank=256, alpha=1.0, train=False):
    lora_layers = nn.ModuleList()

    # Freeze original model
    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        if name.endswith("attn1") or name.endswith("attn2"):
            attn = module

            for proj_name in ['to_q', 'to_k', 'to_v']:
                orig_layer = getattr(attn, proj_name)
                lora_layer = LoRALinear(orig_layer, rank=lora_rank, alpha=alpha)
                setattr(attn, proj_name, lora_layer)
                lora_layers.append(lora_layer)

            if isinstance(attn.to_out, (nn.ModuleList, nn.Sequential)):
                orig_out = attn.to_out[0]
                lora_out = LoRALinear(orig_out, rank=lora_rank, alpha=alpha)
                attn.to_out[0] = lora_out
                lora_layers.append(lora_out)

    if train:
        for lora_layer in lora_layers:
            for param in lora_layer.parameters():
                param.requires_grad = True

    return lora_layers


def inject_lora_adapter(model: nn.Module, lora_path: str, device: torch.device):
    """
    Load LoRA adapter weights into a model that has already been wrapped
    with LoRALinear layers using `init_lora_attn()`.
    Args:
        model (nn.Module): Model with LoRALinear layers already injected.
        lora_path (str): Path to the saved LoRA weights (.pth or .bin).
        device (torch.device): Target device for the weights.
    Raises:
        RuntimeError: If no LoRALinear layers are found in the model.
    """
    # Collect all LoRALinear layer names
    lora_layer_names = [
        name for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]

    if not lora_layer_names:
        raise RuntimeError("No LoRALinear layers found. Did you run init_lora_attn()?")

    # Load saved LoRA state dict
    lora_state_dict = torch.load(lora_path, map_location=device)

    # Filter out keys that match LoRALinear parameters
    filtered_state_dict = {
        k: v for k, v in lora_state_dict.items()
        if any(layer_name in k for layer_name in lora_layer_names)
    }

    if not filtered_state_dict:
        raise RuntimeError(f"No matching LoRA parameters found in {lora_path}")

    # Load LoRA weights (non-strict to avoid missing base weights)
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)

    print(f"[INFO] Loaded LoRA weights from {lora_path}")
    if missing:
        print(f"[WARNING] Missing keys (likely frozen base weights): {missing}")
    if unexpected:
        print(f"[WARNING] Unexpected keys in LoRA state dict: {unexpected}")

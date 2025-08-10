import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA-augmented linear layer for parameter-efficient fine-tuning.

    This wraps an existing nn.Linear layer, freezing its original weights
    and adding a trainable low-rank update using two matrices (lora_A and lora_B).

    The update is computed as:
        ΔW = (lora_B @ lora_A) * (alpha / rank)

    During the forward pass:
        output = original(x) + ΔW(x)

    Args:
        original_linear (nn.Linear):
            The pretrained linear layer to augment with LoRA.
        rank (int, optional):
            The rank 'r' of the low-rank approximation. Defaults to 512.
        alpha (float, optional):
            Scaling factor for the LoRA update. Defaults to 1.0.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 512, alpha: float = 1.0) -> None:
        super().__init__()

        if not isinstance(original_linear, nn.Linear):
            raise TypeError(
                f"LoRALinear only supports nn.Linear, but got: {type(original_linear)}"
            )

        self.original: nn.Linear = original_linear
        self.rank: int = rank
        self.alpha: float = alpha
        self.scaling: float = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA trainable parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, rank) * 0.01)

        # Freeze original parameters
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Method.
        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor:
                Output tensor of shape (batch_size, out_features).
        """
        return self.original(x) + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)


def init_lora_attn(model: nn.Module, lora_rank: int = 512, alpha: float = 1.0) -> nn.ModuleList:
    """
    Wraps LoRA adapters around attention projection layers in a model and makes them trainable.

    This function:
    1. Freezes all original model parameters.
    2. Finds attention modules whose names end in 'attn1' or 'attn2'.
    3. Replaces their projection layers (`to_q`, `to_k`, `to_v`,
       and the first `to_out` layer) with LoRA-augmented versions.

    Args:
        model (nn.Module):
            The base model containing attention modules.
        lora_rank (int, optional):
            Rank of the LoRA decomposition. Defaults to 512.
        alpha (float, optional):
            Scaling factor for LoRA updates. Defaults to 1.0.

    Returns:
        nn.ModuleList:
            List of all LoRA layers injected into the model.
    """
    lora_layers = nn.ModuleList()

    # Safety Check: freeze everything first so base weights never train accidentally
    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        if name.endswith("attn1") or name.endswith("attn2"):
            attn = module

            # Wrap projections: to_q, to_k, to_v
            for proj_name in ['to_q', 'to_k', 'to_v']:
                orig_layer = getattr(attn, proj_name)
                lora_layer = LoRALinear(orig_layer, rank=lora_rank, alpha=alpha)
                setattr(attn, proj_name, lora_layer)
                lora_layers.append(lora_layer)

            # Wrap to_out[0] if it's a list or sequential - case with diffusers 
            if isinstance(attn.to_out, (nn.ModuleList, nn.Sequential)):
                orig_out = attn.to_out[0]
                lora_out = LoRALinear(orig_out, rank=lora_rank, alpha=alpha)
                attn.to_out[0] = lora_out
                lora_layers.append(lora_out)

    return lora_layers

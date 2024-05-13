import torch
import torch.nn as nn
from timm.models.nfnet import NfCfg, NormFreeNet
from timm.models.vision_transformer import VisionTransformer


class ConvNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = NormFreeNet(
            NfCfg(
                depths=(3, 4, 6, 3),
                channels=(256, 512, 1024, 2048),
                stem_chs=64,
                bottle_ratio=0.25,
                group_size=None,
                act_layer="relu",
                attn_layer=None,
            ),
            num_classes=100,
        )

    def count_params(self):
        param_sum = sum([p.numel() for _, p in self.model.named_parameters()])
        print(f"Total params: {param_sum/1000000:.2f} M")
        return param_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits


class ViT(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = VisionTransformer(
            img_size=32,
            patch_size=4,
            embed_dim=384,
            depth=12,
            num_heads=6,
            num_classes=100,
        )

    def count_params(self):
        param_sum = sum([p.numel() for _, p in self.model.named_parameters()])
        print(f"Total params: {param_sum/1000000:.2f} M")
        return param_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits

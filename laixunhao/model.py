import torch
import torch.nn as nn
import timm


class ConvNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = timm.create_model(
            "dm_nfnet_f0.dm_in1k", pretrained=True, num_classes=100
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
        self.model = timm.create_model(
            "vit_base_patch8_224.augreg_in21k_ft_in1k",
            pretrained=True,
            num_classes=100,
            img_size=(32, 32),
        )

    def count_params(self):
        param_sum = sum([p.numel() for _, p in self.model.named_parameters()])
        print(f"Total params: {param_sum/1000000:.2f} M")
        return param_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits

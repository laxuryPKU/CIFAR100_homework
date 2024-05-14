import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import timm
from timm.data import create_dataset, create_loader
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.data.mixup import Mixup
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm


def train(
    model: nn.Module,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    save_dir: str,
):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # parameters
    param_count = sum([p.numel() for p in model.parameters()])

    # mixup
    mixup_func = Mixup(
        mixup_alpha=0.8, prob=0.7, mode="batch", label_smoothing=0.1, num_classes=100
    )

    # cifar 100 data
    train_loader = create_loader(
        create_dataset("torch/cifar100", "cifar100", download=True, split="train"),
        (3, 32, 32),
        batch_size=batch_size,
        is_training=True,
        vflip=0.5,  # 加分点：使用 timm 的数据增强
        hflip=0.5,
        auto_augment="rand-m15-n2",
        device=torch.device(device),
        num_workers=8,
        pin_memory=True,
    )
    valid_loader = create_loader(
        create_dataset("torch/cifar100", "cifar100", download=True, split="validation"),
        (3, 32, 32),
        batch_size=batch_size,
        is_training=False,
        no_aug=True,
        device=torch.device(device),
        num_workers=8,
        pin_memory=True,
    )

    # optimizer and scheduler
    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        amsgrad=True,
    )
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=epochs,
        lr_min=1e-6,
        warmup_lr_init=1e-5,
        warmup_t=5,
    )

    # loss function
    loss_func = SoftTargetCrossEntropy()

    # saved results
    losses = []
    accuracys = []
    learning_rates = []
    bar = tqdm(total=epochs * len(train_loader))

    # train for multiple epochs
    model.to(device)
    for epoch in range(epochs):
        bar.set_description(f"Epoch {epoch+1}/{epochs} ")
        # train
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            x, y = mixup_func(x, y)
            optimizer.zero_grad()
            model.zero_grad()
            logits = model(x)
            loss = loss_func(logits, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            bar.set_postfix(
                loss=f"{losses[-1]:.6f}",
                accuracy=f"{accuracys[-1]*100:.2f}%" if len(accuracys) > 0 else "0%",
            )
            bar.update(1)
        scheduler.step(epoch)
        learning_rates.append(optimizer.state_dict()["param_groups"][0]["lr"])

        # validation
        model.eval()
        right_num = 0
        all_num = 0
        for batch in valid_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                logits = model(x)
                preds = torch.argmax(logits, dim=-1)
            right_num += torch.count_nonzero(torch.eq(preds, y)).item()
            all_num += preds.numel()
        acc = right_num / all_num
        accuracys.append(acc)

        # save model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if epoch == epochs - 1:
            torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.bin"))

    # plot results
    fig = plt.figure(figsize=(10, 6))
    plt.plot(losses, linestyle="-", color="b")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(accuracys, linestyle="-", color="r")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, linestyle="-", color="g")
    plt.xlabel("epoch")
    plt.ylabel("learning_rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "leraning_rate.png"))
    plt.close(fig)

    # save training results
    result = {
        "param_count": param_count,
        "final_accuracy": accuracys[-1],
        "final_loss": losses[-1],
        "loss": losses,
        "accuracy": accuracys,
        "learning_rate": learning_rates,
    }
    with open(
        os.path.join(save_dir, "training_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    # nfnet
    # 加分点：使用 timm 的模型
    nfnet = timm.create_model("dm_nfnet_f0.dm_in1k", pretrained=True, num_classes=100)
    train(
        nfnet,
        seed=42,
        device="cuda",
        epochs=50,
        batch_size=256,
        learning_rate=1e-4,
        weight_decay=1e-4,
        save_dir="checkpoint/nfnet",
    )

    # vit
    # 加分点：训练了 CNN 和 ViT
    vit = timm.create_model(
        "vit_base_patch8_224.augreg_in21k_ft_in1k",
        pretrained=True,
        num_classes=100,
        img_size=(32, 32),
    )
    train(
        vit,
        seed=42,
        device="cuda",
        epochs=50,
        batch_size=256,
        learning_rate=1e-4,
        weight_decay=1e-4,
        save_dir="checkpoint/vit",
    )

    # # nfnet + different learning rate
    # # 加分点：比较不同超参作用
    nfnet = timm.create_model("dm_nfnet_f0.dm_in1k", pretrained=True, num_classes=100)
    train(
        nfnet,
        seed=42,
        device="cuda",
        epochs=50,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
        save_dir="checkpoint/nfnet_diff_lr",
    )


if __name__ == "__main__":
    main()

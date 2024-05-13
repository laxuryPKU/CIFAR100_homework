import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import timm
import timm.data
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
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

    # cifar 100 data
    train_loader = timm.data.create_loader(
        timm.data.create_dataset(
            "torch/cifar100", "cifar100", download=True, split="train"
        ),
        (3, 32, 32),
        batch_size=batch_size,
        is_training=True,
        vflip=0.5,  # 加分点：使用 timm 的数据增强
        hflip=0.5,
        device=torch.device(device),
        num_workers=8,
        pin_memory=True,
    )
    valid_loader = timm.data.create_loader(
        timm.data.create_dataset(
            "torch/cifar100", "cifar100", download=True, split="validation"
        ),
        (3, 32, 32),
        batch_size=batch_size,
        is_training=False,
        no_aug=True,
        device=torch.device(device),
        num_workers=8,
        pin_memory=True,
    )

    # optimizer and scheduler
    optimizer = Adam(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs * math.ceil(50000 / batch_size),
        eta_min=1e-6,
    )

    # loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # saved results
    losses = []
    accuracys = []
    bar = tqdm(total=epochs * len(train_loader))

    # train for multiple epochs
    model.to(device)
    for epoch in range(epochs):
        bar.set_description(f"Epoch {epoch+1}/{epochs} ")
        # train
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
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
        scheduler.step()

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
    plt.xlabel("iter")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy.png"))
    plt.close(fig)

    # save training results
    result = {
        "param_count": param_count,
        "final_accuracy": accuracys[-1],
        "final_loss": losses[-1],
        "loss": losses,
        "accuracy": accuracys,
    }
    with open(
        os.path.join(save_dir, "training_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    # resnet
    # 加分点：使用 timm 的模型
    resnet = timm.create_model(
        "resnetv2_50x1_bit.goog_distilled_in1k", pretrained=False, num_classes=100
    )
    train(
        resnet,
        seed=42,
        device="cuda:0",
        epochs=100,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
        save_dir="checkpoint/resnet50",
    )

    # vit
    # 加分点：训练了 CNN 和 ViT
    vit = timm.create_model(
        "tiny_vit_21m_224.dist_in22k_ft_in1k", pretrained=False, num_classes=100
    )
    train(
        vit,
        seed=42,
        device="cuda:0",
        epochs=100,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
        save_dir="checkpoint/vit",
    )

    # # resnet + different learning rate
    # # 加分点：比较不同超参作用
    resnet = timm.create_model(
        "resnetv2_50x1_bit.goog_distilled_in1k", pretrained=False, num_classes=100
    )
    train(
        resnet,
        seed=42,
        device="cuda:0",
        epochs=100,
        batch_size=256,
        learning_rate=1e-2,
        weight_decay=1e-4,
        save_dir="checkpoint/resnet50_diff_lr",
    )


if __name__ == "__main__":
    main()

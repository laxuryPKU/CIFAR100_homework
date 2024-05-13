import math
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import timm
import timm.data
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from tqdm import tqdm


@dataclass
class TrainingArguments:
    seed: int = 114514
    device: str = "cpu"
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 128
    checkpoint: str = None
    save_epochs: int = -1
    save_dir: str = "checkpoint"
    optimizer: str = "adamw"
    scheduler: str = "cosine"


def get_cifar100_loader(args: TrainingArguments):
    train_set = timm.data.create_dataset(
        "torch/cifar100", "cifar100", download=True, split="train"
    )
    train_loader = timm.data.create_loader(
        train_set,
        (3, 32, 32),
        batch_size=args.batch_size,
        is_training=True,
        vflip=0.5,
        hflip=0.5,
        device=torch.device(args.device),
        num_workers=8,
        pin_memory=True,
    )
    valid_set = timm.data.create_dataset(
        "torch/cifar100", "cifar100", download=True, split="validation"
    )
    valid_loader = timm.data.create_loader(
        valid_set,
        (3, 32, 32),
        batch_size=args.batch_size,
        is_training=False,
        no_aug=True,
        device=torch.device(args.device),
        num_workers=8,
        pin_memory=True,
    )
    return train_loader, valid_loader


def get_optimizer_scheduler(model: nn.Module, args: TrainingArguments):
    if args.optimizer == "adamw":
        optimizer = AdamW(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "sgdm":
        optimizer = SGD(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=0.9,
        )
    else:
        raise NotImplementedError
    total_steps = args.epochs * math.ceil(50000 / args.batch_size)
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer=optimizer, T_max=total_steps, eta_min=1e-6
        )
    elif args.scheduler == "linear":
        scheduler = LinearLR(
            optimizer=optimizer,
            total_iters=total_steps,
            start_factor=1,
            end_factor=0.001,
        )
    else:
        raise NotImplementedError
    return optimizer, scheduler


def train(model: nn.Module, args: TrainingArguments):
    # data loader
    train_loader, valid_loader = get_cifar100_loader(args)

    # optimizer and scheduler
    optimizer, scheduler = get_optimizer_scheduler(model, args)

    # loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # saved results
    loss_list = []
    acc_list = []
    total_steps = args.epochs * len(train_loader)
    bar = tqdm(total=total_steps)

    # train for multiple epochs
    model.to(args.device)
    for epoch in range(args.epochs):
        bar.set_description(f"Epoch {epoch+1}/{args.epochs} ")
        # train
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(args.device), batch[1].to(args.device)
            optimizer.zero_grad()
            model.zero_grad()
            logits = model(x)
            loss = loss_func(logits, y)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            bar.set_postfix(
                loss=f"{loss_list[-1]:.6f}",
                accuracy=f"{acc_list[-1]*100:.2f}%" if len(acc_list) > 0 else "0%",
            )
            bar.update(1)
        scheduler.step()

        # validation
        model.eval()
        right_num = 0
        all_num = 0
        for batch in valid_loader:
            x, y = batch[0].to(args.device), batch[1].to(args.device)
            with torch.no_grad():
                logits = model(x)
                preds = torch.argmax(logits, dim=-1)
            right_num += torch.count_nonzero(torch.eq(preds, y)).item()
            all_num += preds.numel()
        acc = right_num / all_num
        acc_list.append(acc)

        # save model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)
        if (
            args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0
        ) or epoch == args.epochs - 1:
            torch.save(
                model.state_dict(), os.path.join(args.save_dir, f"epoch_{epoch}.bin")
            )

    return model, loss_list, acc_list


def test(model: nn.Module, args: TrainingArguments):
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict)
    _, valid_loader = get_cifar100_loader(args)

    model.to(args.device)
    model.eval()
    right_num = 0
    all_num = 0
    for batch in valid_loader:
        x, y = batch[0].to(args.device), batch[1].to(args.device)
        with torch.no_grad():
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
        right_num += torch.count_nonzero(torch.eq(preds, y)).item()
        all_num += preds.numel()
    acc = right_num / all_num
    return acc


def plot_result(result_list: list, x_label: str, y_label, save_path: str):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(result_list, linestyle="-", color="b")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, f"{y_label}.png"))
    plt.close(fig)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

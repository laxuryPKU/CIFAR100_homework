import json
import os

import torch.nn as nn
from model import ConvNet, ViT
from train import TrainingArguments, plot_result, set_seed, test, train


def main(model: nn.Module, args: TrainingArguments):
    set_seed(args.seed)
    param_count = model.count_params()
    model, loss_list, acc_list = train(model, args)
    plot_result(loss_list, "steps", "loss", args.save_dir)
    plot_result(acc_list, "epochs", "accuracy", args.save_dir)
    test_acc = test(model, args)
    print(f"========= Final Accuracy {test_acc*100:.2f} =========")

    result = {
        "param_count": param_count,
        "final_accuracy": test_acc,
        "loss_list": loss_list,
        "accuracy_list": acc_list,
    }
    with open(os.path.join(args.save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # convnet+adamw
    main(
        ConvNet(),
        TrainingArguments(
            device="cuda:0",
            epochs=100,
            learning_rate=0.001,
            weight_decay=0.01,
            batch_size=256,
            checkpoint=None,
            save_dir="checkpoint/convnet/adamw",
            optimizer="adamw",
            scheduler="cosine",
        ),
    )

    # convnet+sgdm
    main(
        ConvNet(),
        TrainingArguments(
            device="cuda:0",
            epochs=100,
            learning_rate=0.001,
            weight_decay=0.01,
            batch_size=256,
            checkpoint=None,
            save_dir="checkpoint/convnet/sgdm",
            optimizer="sgdm",
            scheduler="cosine",
        ),
    )

    # convnet+adamw+linear
    main(
        ConvNet(),
        TrainingArguments(
            device="cuda:0",
            epochs=100,
            learning_rate=0.001,
            weight_decay=0.01,
            batch_size=256,
            checkpoint=None,
            save_dir="checkpoint/convnet/linear_decay",
            optimizer="adamw",
            scheduler="linear",
        ),
    )

    # vit
    main(
        ViT(),
        TrainingArguments(
            device="cuda:0",
            epochs=100,
            learning_rate=0.001,
            weight_decay=0.01,
            batch_size=256,
            checkpoint=None,
            save_dir="checkpoint/ViT",
            optimizer="adamw",
            scheduler="cosine",
        ),
    )

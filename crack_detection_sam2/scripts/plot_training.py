"""畫訓練曲線: loss / pixel_accuracy / mIoU / per-class IoU。

讀取 train.py 寫的 log.json (含 args + history),
輸出 PNG 到同一個 output_dir。

用法:
    python scripts/plot_training.py --log_dir outputs/stem_fold0_small_2class_v2
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(log_path):
    with open(log_path) as f:
        log = json.load(f)
    hist = log.get("history", [])
    epochs = [h["epoch"] for h in hist]
    train_loss = [h["train"]["loss"] for h in hist]
    train_ce = [h["train"]["ce"] for h in hist]
    train_dice = [h["train"]["dice"] for h in hist]

    val_ce = [h["val"].get("val_ce_loss", float("nan")) for h in hist]
    val_miou = [h["val"].get("miou", float("nan")) for h in hist]
    val_mdice = [h["val"].get("mdice", float("nan")) for h in hist]
    val_pacc = [h["val"].get("pixel_accuracy", float("nan")) for h in hist]

    per_class_iou = {}
    for h in hist:
        per = h["val"].get("per_class", {})
        for name, m in per.items():
            per_class_iou.setdefault(name, []).append(m["iou"])
    return {
        "args": log.get("args", {}),
        "epochs": epochs,
        "train_loss": train_loss, "train_ce": train_ce, "train_dice": train_dice,
        "val_ce": val_ce, "val_miou": val_miou, "val_mdice": val_mdice,
        "val_pacc": val_pacc, "per_class_iou": per_class_iou,
    }


def plot_all(log_dir, log):
    epochs = log["epochs"]
    if not epochs:
        print("history 為空, 還沒跑完第一個 epoch")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(epochs, log["train_loss"], label="train total")
    ax.plot(epochs, log["train_ce"], label="train CE", linestyle="--")
    ax.plot(epochs, log["train_dice"], label="train Dice", linestyle=":")
    ax.plot(epochs, log["val_ce"], label="val CE", color="tab:red")
    ax.set_title("loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, log["val_pacc"], label="pixel_accuracy", color="tab:green")
    ax.set_title("pixel accuracy (val)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, log["val_miou"], label="mIoU", color="tab:purple")
    ax.plot(epochs, log["val_mdice"], label="mDice", color="tab:orange")
    ax.set_title("macro mIoU / mDice (val, no bg)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    for name, vals in log["per_class_iou"].items():
        ax.plot(epochs[:len(vals)], vals, label=name)
    ax.set_title("per-class IoU (val)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(os.path.basename(log_dir.rstrip("/")) +
                 f"  | best mIoU={max(log['val_miou']):.4f}"
                 f" @ ep{1 + log['val_miou'].index(max(log['val_miou']))}")
    fig.tight_layout()
    out = os.path.join(log_dir, "training_curves.png")
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True,
                        help="train.py 的 output_dir, 內含 log.json")
    args = parser.parse_args()

    log_path = os.path.join(args.log_dir, "log.json")
    if not os.path.isfile(log_path):
        raise SystemExit(f"找不到 {log_path}")
    log = load_log(log_path)
    plot_all(args.log_dir, log)


if __name__ == "__main__":
    main()

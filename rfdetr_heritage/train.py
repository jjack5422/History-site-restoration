import argparse
from pathlib import Path

from rfdetr import RFDETRSegSmall


DATASET_DIR = "/home/zzz90/research/crack_detection/data/merged_4class_rfdetr_coco"
OUTPUT_DIR = "/home/zzz90/research/rfdetr_heritage/runs/seg_small"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    model = RFDETRSegSmall()

    model.train(
        dataset_dir=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        resolution=args.resolution,
        num_workers=args.num_workers,
        resume=args.resume,
        early_stopping=True,
        tensorboard=True,
    )


if __name__ == "__main__":
    main()

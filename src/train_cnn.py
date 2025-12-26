from __future__ import annotations

import argparse
from .data import load_sign_mnist_csv
from .models import build_cnn
from .train_common import TrainConfig, compile_model, train_and_evaluate

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/raw")
    p.add_argument("--out_dir", type=str, default="artifacts/cnn")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    ds = load_sign_mnist_csv(args.data_dir)
    model = compile_model(build_cnn(), lr=args.lr, weight_decay=args.weight_decay)

    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    _, metrics = train_and_evaluate(model, ds, cfg)
    print(f"[CNN] test_accuracy={metrics['test_accuracy']:.4f}  saved={metrics['export_dir']}")

if __name__ == "__main__":
    main()

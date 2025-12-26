from __future__ import annotations

import json
import os
from typing import Dict, Any

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_history_plot(history, out_prefix: str) -> None:
    if plt is None:
        return
    h = history.history
    if not h:
        return

    if "loss" in h:
        plt.figure()
        plt.plot(h["loss"], label="train_loss")
        if "val_loss" in h:
            plt.plot(h["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + "_loss.png", dpi=150)
        plt.close()

    if "accuracy" in h:
        plt.figure()
        plt.plot(h["accuracy"], label="train_acc")
        if "val_accuracy" in h:
            plt.plot(h["val_accuracy"], label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + "_acc.png", dpi=150)
        plt.close()

def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

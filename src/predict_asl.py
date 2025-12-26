from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


def load_classes(model_dir: str) -> list[str]:
    # 兼容：优先读取同目录 meta.json
    meta_path = os.path.join(os.path.dirname(model_dir), "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta["classes"]
    # fallback：没有 meta.json 就用目录名（不推荐）
    raise FileNotFoundError(f"meta.json not found next to model: {meta_path}")


def preprocess_rgb_raw255(path: str, h: int, w: int) -> np.ndarray:
    """关键点：保持 raw_0_255，不做 /255，不做 preprocess_input。"""
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    x = np.asarray(img).astype("float32")  # 0~255
    return x


def topk(prob: np.ndarray, classes: list[str], k: int = 5) -> list[Tuple[str, float]]:
    idx = np.argsort(prob)[::-1][:k]
    return [(classes[int(i)], float(prob[int(i)])) for i in idx]


def list_images(image_dir: str, exts: List[str]) -> List[str]:
    exts = [e.lower().lstrip(".") for e in exts]
    out = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            ext = f.split(".")[-1].lower()
            if ext in exts:
                out.append(os.path.join(root, f))
    return sorted(out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="best.keras 或 saved_model 目录")
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--image_dir", type=str, default=None)
    p.add_argument("--ext", type=str, nargs="+", default=["jpg", "jpeg", "png"])
    p.add_argument("--topk", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    # load model (Keras3 兼容 SavedModel)
    model_path = args.model
    if tf.io.gfile.isdir(model_path):
        from keras.layers import TFSMLayer
        layer = TFSMLayer(model_path, call_endpoint="serve")
        model = tf.keras.Sequential([layer])
        # meta.json 在 saved_model 上一级
        classes = load_classes(os.path.join(os.path.dirname(model_path), "best.keras"))
    else:
        model = tf.keras.models.load_model(model_path)
        classes = load_classes(model_path)

    h, w = model.input_shape[1], model.input_shape[2]

    def predict(path: str):
        x = preprocess_rgb_raw255(path, h, w)
        prob = model.predict(x[None, ...], verbose=0)[0]
        pred = classes[int(np.argmax(prob))]
        print(f"{os.path.basename(path):<14s} -> {pred}")
        for c, p in topk(prob, classes, args.topk):
            print(f"  {c:<8s} {p:.4f}")

    if args.image:
        predict(args.image)
        return
    if args.image_dir:
        for pth in list_images(args.image_dir, args.ext):
            predict(pth)
        return
    print("请提供 --image 或 --image_dir")


if __name__ == "__main__":
    main()

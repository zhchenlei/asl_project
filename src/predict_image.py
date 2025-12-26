from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
from PIL import Image

from .data import IDX2CHAR


def preprocess_image(path: str) -> np.ndarray:
    # 转灰度 -> 居中裁剪正方形 -> resize 到 28x28 -> 归一化到 [0,1]
    img = Image.open(path).convert("L")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((28, 28), Image.BILINEAR)

    arr = np.asarray(img).astype("float32") / 255.0
    # 自适应反相：均值偏亮时反相，让“手势区域”更偏亮（与训练数据更一致）
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    return arr.reshape((28, 28, 1))


def predict_one(model: tf.keras.Model, img_arr: np.ndarray) -> Tuple[str, float]:
    pred = model.predict(img_arr[None, ...], verbose=0)[0]
    idx = int(np.argmax(pred))
    return IDX2CHAR[idx], float(pred[idx])


def list_images(image_dir: str, exts: List[str]) -> List[str]:
    exts = [e.lower().lstrip(".") for e in exts]
    paths: List[str] = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            ext = f.split(".")[-1].lower()
            if ext in exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True, help="saved_model 目录")
    p.add_argument("--image", type=str, default=None, help="单张图片路径")
    p.add_argument("--image_dir", type=str, default=None, help="图片文件夹")
    p.add_argument("--ext", type=str, nargs="+", default=["jpg", "jpeg", "png"])
    return p.parse_args()


def main():
    args = parse_args()
    model_path = args.model_dir
    # Keras 3: load_model 仅支持 .keras/.h5；SavedModel 需用 TFSMLayer (inference-only)
    if tf.io.gfile.isdir(model_path):
        # 训练日志显示 endpoint 名称为 'serve'
        layer = TFSMLayer(model_path, call_endpoint='serve')
        model = tf.keras.Sequential([layer])
    else:
        model = tf.keras.models.load_model(model_path)


    if args.image:
        arr = preprocess_image(args.image)
        c, p = predict_one(model, arr)
        print(f"{args.image} -> {c}  prob={p:.4f}")
        return

    if args.image_dir:
        imgs = list_images(args.image_dir, args.ext)
        if not imgs:
            print("未找到图片。")
            return
        for path in imgs:
            arr = preprocess_image(path)
            c, p = predict_one(model, arr)
            print(f"{path} -> {c}  prob={p:.4f}")
        return

    print("请提供 --image 或 --image_dir")


if __name__ == "__main__":
    main()

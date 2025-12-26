# src/train_transfer.py
from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", type=str, default="data/asl_alphabet_train")
    p.add_argument("--test_dir", type=str, default="data/asl_alphabet_test")
    p.add_argument("--out_dir", type=str, default="artifacts/transfer")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fine_tune", action="store_true", help="二阶段微调（解冻部分 backbone）")
    p.add_argument("--weights_path", type=str, default=None, help="本地 ImageNet 权重 .h5（可选，避免联网下载）")
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    tf.keras.utils.set_random_seed(args.seed)

    IMG_SIZE = (args.img_size, args.img_size)

    # 训练/验证集：从 train_dir 自动按子文件夹取类别
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.train_dir,
        validation_split=0.1,
        subset="training",
        seed=args.seed,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.train_dir,
        validation_split=0.1,
        subset="validation",
        seed=args.seed,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        label_mode="int",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Num classes:", num_classes)

    # 加速
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    # 轻量数据增强（真实照片必备）
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomContrast(0.1),
    ])

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=None,  # 避免运行时联网下载
    )
    # 若提供本地权重则加载（ImageNet no_top）
    if args.weights_path:
        print("Loading weights from:", args.weights_path)
        base.load_weights(args.weights_path)
    else:
        print("WARNING: weights_path not provided. Training from scratch (slower, worse).")
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = aug(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = os.path.join(args.out_dir, "best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    print("\n[Stage-1] Train head only ...")
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)

    hist_path = os.path.join(args.out_dir, "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist1.history, f, ensure_ascii=False, indent=2)
    print("Saved history:", hist_path)


    # 保存 history（用于画训练曲线图）

    hist_path = os.path.join(args.out_dir, "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist1.history, f, ensure_ascii=False, indent=2)
    print("Saved history:", hist_path)


    # 可选：二阶段微调（解冻最后一部分）
    if args.fine_tune:
        print("\n[Stage-2] Fine-tune backbone last layers ...")
        base.trainable = True
        # 只解冻后 30 层（经验值，兼顾效果与稳定）
        for layer in base.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr * 0.1),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train_ds, validation_data=val_ds, epochs=max(5, args.epochs // 2), callbacks=callbacks, verbose=2)

    # 保存类别映射（推理必须用）
    meta = {
        "img_size": args.img_size,
        "classes": class_names,
        "num_classes": num_classes,
        "model": "MobileNetV2",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 导出 SavedModel（部署用）
    export_dir = os.path.join(args.out_dir, "saved_model")
    model.export(export_dir)
    print("Saved:", ckpt_path)
    print("Exported:", export_dir)

    # 额外：用 test_dir（每类1张）做一次简单 sanity check（不是正式指标）
    if os.path.isdir(args.test_dir):
        test_files = sorted([p for p in Path(args.test_dir).glob("*_test.jpg")])
        if test_files:
            print("\n[Sanity-check] predict asl_alphabet_test ...")
            ok = 0
            for fp in test_files:
                true = fp.name.split("_test.jpg")[0]
                img = tf.keras.utils.load_img(fp, target_size=IMG_SIZE)
                arr = tf.keras.utils.img_to_array(img)[None, ...]
                prob = model.predict(arr, verbose=0)[0]
                pred = class_names[int(np.argmax(prob))]
                ok += int(pred.lower() == true.lower())
                print(f"{fp.name:15s}  GT={true:7s}  PR={pred:7s}  top1={float(prob.max()):.3f}")
            print(f"Sanity-check acc: {ok}/{len(test_files)} = {ok/len(test_files):.3f}")
        else:
            print("No *_test.jpg found in test_dir; skip sanity-check.")


if __name__ == "__main__":
    main()

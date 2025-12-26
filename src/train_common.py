from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .utils import ensure_dir, save_history_plot, save_json
from .data import IDX2CHAR

@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    epochs: int = 15
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42

def compile_model(model: tf.keras.Model, lr: float, weight_decay: float) -> tf.keras.Model:
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_and_evaluate(model: tf.keras.Model, dataset, cfg: TrainConfig) -> Tuple[tf.keras.Model, dict]:
    ensure_dir(cfg.out_dir)
    tf.keras.utils.set_random_seed(cfg.seed)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cfg.out_dir, "best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        dataset.x_train, dataset.y_train,
        validation_split=0.1,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
    y_pred = np.argmax(model.predict(dataset.x_test, batch_size=cfg.batch_size, verbose=0), axis=1)

    report = classification_report(dataset.y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(dataset.y_test, y_pred)

    save_history_plot(history, os.path.join(cfg.out_dir, "training"))
    save_json(
        {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "labels": IDX2CHAR,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        },
        os.path.join(cfg.out_dir, "report.json"),
    )

    export_dir = os.path.join(cfg.out_dir, "saved_model")
    model.export(export_dir)
    return model, {"test_loss": test_loss, "test_accuracy": test_acc, "export_dir": export_dir}

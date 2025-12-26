import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL = "artifacts/transfer_step15/best.keras"
META  = "artifacts/transfer_step15/meta.json"
TEST_DIR = "data/asl_test_more_50"   # 确保这里是【按类别分文件夹】

model = tf.keras.models.load_model(MODEL)
classes = json.load(open(META, "r", encoding="utf-8"))["classes"]

H, W = model.input_shape[1], model.input_shape[2]

stats = {
    c: {"total": 0, "correct": 0, "conf_sum": 0.0}
    for c in classes
}

def preprocess(path):
    img = Image.open(path).convert("RGB").resize((W, H))
    x = np.asarray(img).astype("float32")
    return x[None, ...]

print("🔍 Start scanning test dataset...")

found_any = False

for cls in sorted(os.listdir(TEST_DIR)):
    cls_dir = os.path.join(TEST_DIR, cls)
    if not os.path.isdir(cls_dir):
        continue
    if cls not in classes:
        print(f"⚠️ Skip unknown class folder: {cls}")
        continue

    imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg")]
    print(f"  → {cls}: {len(imgs)} images")

    for fn in imgs:
        found_any = True
        x = preprocess(os.path.join(cls_dir, fn))
        prob = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(prob))
        pred_cls = classes[pred_idx]
        pred_conf = float(prob[pred_idx])

        stats[cls]["total"] += 1
        if pred_cls == cls:
            stats[cls]["correct"] += 1
            stats[cls]["conf_sum"] += pred_conf

if not found_any:
    print("❌ ERROR: No valid test images found. Check TEST_DIR.")
    exit(1)

print("\n=== Per-Class Accuracy & Confidence ===")
print(f"{'Class':<10} {'Acc':>8} {'AvgConf':>10} {'Samples':>8}")
print("-" * 42)

for cls in classes:
    total = stats[cls]["total"]
    if total == 0:
        continue
    correct = stats[cls]["correct"]
    acc = correct / total
    avg_conf = stats[cls]["conf_sum"] / correct if correct > 0 else 0.0

    print(f"{cls:<10} {acc:>8.3f} {avg_conf:>10.3f} {total:>8}")

print("✅ Done.")

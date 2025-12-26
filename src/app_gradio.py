import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

# 你的类别映射（与训练一致）
from src.data import IDX2CHAR

# 优先加载 best.keras（Keras3 兼容最好）
MODEL_PATH = "artifacts/cnn/best.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_pil(img: Image.Image) -> np.ndarray:
    # 与 src/predict_image.py 保持一致（但默认不做“自动反相”，更稳定）
    img = img.convert("L")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((28, 28), Image.BILINEAR)

    arr = (np.asarray(img).astype("float32") / 255.0)

    # 可选：如果你要尝试反相，把下面两行取消注释
    # if arr.mean() > 0.5:
    #     arr = 1.0 - arr

    return arr.reshape(1, 28, 28, 1)

def predict(img: Image.Image):
    x = preprocess_pil(img)
    prob = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(prob))
    letter = IDX2CHAR[str(idx)] if isinstance(IDX2CHAR, dict) and "0" in IDX2CHAR else IDX2CHAR[idx]
    return letter, float(prob[idx])

def predict_with_scores(img: Image.Image):
    x = preprocess_pil(img)
    prob = model.predict(x, verbose=0)[0]
    # 返回 Top-5
    top = np.argsort(prob)[::-1][:5]
    mapping = {}
    for i in top:
        letter = IDX2CHAR[str(int(i))] if isinstance(IDX2CHAR, dict) and "0" in IDX2CHAR else IDX2CHAR[int(i)]
        mapping[letter] = float(prob[i])
    best = max(mapping, key=mapping.get)
    return best, mapping

demo = gr.Interface(
    fn=predict_with_scores,
    inputs=gr.Image(type="pil", label="上传/拖拽手势图片"),
    outputs=[
        gr.Textbox(label="预测字母"),
        gr.Label(num_top_classes=5, label="Top-5 概率"),
    ],
    title="美国手语字母识别（CNN）",
    description="上传一张手势图片，模型将输出预测字母与Top-5概率（best.keras）。"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

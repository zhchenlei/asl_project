import json
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

MODEL_PATH = "artifacts/transfer_small/best.keras"
META_PATH  = "artifacts/transfer_small/meta.json"

model = tf.keras.models.load_model(MODEL_PATH)
classes = json.load(open(META_PATH, "r", encoding="utf-8"))["classes"]

# 从模型中自动读取输入尺寸
H, W = model.input_shape[1], model.input_shape[2]

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((W, H))
    x = np.asarray(img).astype("float32")  # 保持 0~255，与训练一致
    return x[None, ...]

def predict(img: Image.Image):
    x = preprocess(img)
    prob = model.predict(x, verbose=0)[0]
    top = np.argsort(prob)[::-1][:5]
    best = classes[int(top[0])]
    return best, {classes[i]: float(prob[i]) for i in top}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传手势图片"),
    outputs=[
        gr.Textbox(label="预测结果"),
        gr.Label(num_top_classes=5, label="Top-5 概率")
    ],
    title="ASL Alphabet 识别（29类）",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

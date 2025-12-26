
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import gradio as gr

# 可选：用 MediaPipe 做手部定位
import mediapipe as mp

MODEL_PATH = "artifacts/transfer_small/best.keras"
META_PATH  = "artifacts/transfer_small/meta.json"

model = tf.keras.models.load_model(MODEL_PATH)
classes = json.load(open(META_PATH, "r", encoding="utf-8"))["classes"]

H, W = model.input_shape[1], model.input_shape[2]

# MediaPipe Hands（静态图）
mp_hands = mp.solutions.hands
_hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def center_square_crop(pil_img: Image.Image) -> Image.Image:
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return pil_img.crop((left, top, left + side, top + side))

def detect_hand_crop(pil_img: Image.Image, pad_ratio: float = 0.20):
    """
    用 MediaPipe 找手部 bbox 并裁剪；找不到返回 None
    """
    img = pil_img.convert("RGB")
    rgb = np.array(img)  # (H,W,3) RGB
    h, w, _ = rgb.shape

    res = _hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None

    lm = res.multi_hand_landmarks[0].landmark
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]

    x1 = max(min(xs), 0.0)
    y1 = max(min(ys), 0.0)
    x2 = min(max(xs), 1.0)
    y2 = min(max(ys), 1.0)

    x1 = int(x1 * w); x2 = int(x2 * w)
    y1 = int(y1 * h); y2 = int(y2 * h)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 5 or bh <= 5:
        return None

    pad = int(max(bw, bh) * pad_ratio)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    return img.crop((x1, y1, x2, y2))

def preprocess(pil_img: Image.Image, mirror: bool):
    """
    返回:
      x: (1,H,W,3) float32, 0~255  (与你训练/验证一致)
      seen: PIL.Image 模型实际看到的图（用于可视化排错）
    """
    img = pil_img.convert("RGB")

    # 1) 镜像（左手/前置摄像头）
    if mirror:
        img = ImageOps.mirror(img)

    # 2) 尝试手部裁剪（找不到就回退居中裁剪）
    hand = detect_hand_crop(img)
    if hand is None:
        hand = center_square_crop(img)

    # 3) 等比缩放 + padding 到模型输入大小（不要拉伸）
    seen = ImageOps.pad(hand, (W, H), method=Image.BICUBIC, color=(255, 255, 255), centering=(0.5, 0.5))

    # 4) raw 0~255（你已验证这是正确的推理输入）
    x = np.asarray(seen).astype("float32")
    return x[None, ...], seen

def predict(pil_img: Image.Image, mirror: bool, conf_thr: float):
    x, seen = preprocess(pil_img, mirror=mirror)
    prob = model.predict(x, verbose=0)[0]

    top = np.argsort(prob)[::-1][:5]
    best_i = int(top[0])
    best_p = float(prob[best_i])
    best = classes[best_i]

    # 置信度阈值：低于阈值就提示不确定
    if best_p < conf_thr:
        best_show = f"不确定（top1={best_p:.2f}，当前预测={best}）"
    else:
        best_show = f"{best}（top1={best_p:.2f}）"

    top5 = {classes[int(i)]: float(prob[int(i)]) for i in top}
    return best_show, top5, seen

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="上传手势图片"),
        gr.Checkbox(label="水平翻转（左手/前置摄像头勾选）", value=False),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.80, step=0.01, label="置信度阈值（低于则提示不确定）"),
    ],
    outputs=[
        gr.Textbox(label="预测结果"),
        gr.Label(num_top_classes=5, label="Top-5 概率"),
        gr.Image(type="pil", label="模型实际看到的裁剪图（最关键，用来排错）"),
    ],
    title="ASL Alphabet 识别（29类）- HandCrop + Mirror + DebugView",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)

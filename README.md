# 美国手语图像识别（ASL CSV）- 可直接在 WSL2 Ubuntu 运行

本项目实现：
- 基于 **ASL CSV 数据集**（28x28 灰度像素 + label 0~25）进行 **26 类字母识别**
- 两种方案：**全连接网络（MLP）** 与 **卷积网络（CNN）**
- 训练 / 测试 / 保存模型
- 对**外部图片**（png/jpg）进行推理（自动转灰度、居中裁剪、缩放到 28x28）

> 说明：本项目默认使用 `sign_mnist_train.csv` 与 `sign_mnist_test.csv`（常见的 ASL 字母 CSV 数据集）。
> 你需要自己把数据放到 `data/raw/` 目录（见下方步骤）。

---

## 1. 环境搭建（WSL2 Ubuntu）

### 1.1 安装系统依赖
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git unzip wget
```

### 1.2 创建虚拟环境并安装依赖
```bash
cd ~/projects
mkdir -p asl && cd asl
# 把本项目文件解压到这里（或 git clone）
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## 2. 数据准备

把以下文件放到：
- `data/raw/sign_mnist_train.csv`
- `data/raw/sign_mnist_test.csv`

CSV 格式要求：
- 第一列为 `label`（0~25 对应 A~Z）
- 其余 784 列为 28x28 灰度像素（0~255）

---

## 3. 快速跑通（训练 + 测试）

### 3.1 训练 MLP（全连接）
```bash
python -m src.train_mlp --data_dir data/raw --out_dir artifacts/mlp --epochs 20
```

### 3.2 训练 CNN（推荐）
```bash
python -m src.train_cnn --data_dir data/raw --out_dir artifacts/cnn --epochs 15
```

训练结束会输出：
- 测试集 accuracy
- 保存模型到 `artifacts/*/saved_model/`
- 保存训练曲线与评估报告到 `artifacts/*/`

---

## 4. 外部图片推理（单张/文件夹）

### 4.1 单张图片
```bash
python -m src.predict_image --model_dir artifacts/cnn/saved_model --image path/to/your.jpg
```

### 4.2 批量图片（文件夹）
```bash
python -m src.predict_image --model_dir artifacts/cnn/saved_model --image_dir path/to/images --ext jpg png jpeg
```


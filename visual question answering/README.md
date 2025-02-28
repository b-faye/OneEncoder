# OneEncoder: A Lightweight Framework for Multimodal Training

OneEncoder is a streamlined framework for aligning multiple modalities (text, image, audio, video) using a progressive training approach. It reduces training costs by aligning new modalities without retraining the entire system, achieving strong results even on small datasets.

## 🚀 Key Features
- **Progressive Modality Alignment:**
  - Step 1: Align image and text with a Universal Projection (UP) module.
  - Step 2: Freeze UP, train an Alignment Layer to integrate audio, video, and more.
- **Efficient and Cost-Effective:** Works well on small paired datasets, outperforming large-scale models with specialized encoders.
- **Flexible Backbone Choices:** Supports various image and text encoders (e.g., ALBERT, BERT, RoBERTa, ViT, DeiT, BeiT).

## 🏁 Quickstart

### Datasets
- **Visual QA:** [DAQUAR](https://www.kaggle.com/datasets/tezansahu/processed-daquar-dataset)

## 📘 Usage
Run Visual QA:
```bash
cd "albert and beit"
python albert_beit.py
```

## 🛠️ Fusion Operations
- **For VQA:** Addition, Scaled Dot Product Attention

## 🧠 Demos
- [Colab Demo](Demo)
- [Hugging Face Spaces](https://huggingface.co/spaces/bilalfaye/OneEncoder-retriever)
- Pretrained Models:
  - Text & Image → [HF Model](https://huggingface.co/bilalfaye/OneEncoder-text-image)
  - Text, Image & Audio → [HF Model](https://huggingface.co/bilalfaye/OneEncoder-text-image-audio)
  - Text, Image & Video → [HF Model](https://huggingface.co/bilalfaye/OneEncoder-text-image-video)

---
🔧 **Default Training Config:** `temperature = 2.5`, fusion via `addition`
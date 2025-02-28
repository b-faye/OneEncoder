# OneEncoder: A Lightweight Framework for Multimodal Training

OneEncoder is a streamlined framework for aligning multiple modalities (text, image, audio, video) using a progressive training approach. It reduces training costs by aligning new modalities without retraining the entire system, achieving strong results even on small datasets.

## 🚀 Key Features
- **Progressive Modality Alignment:**
  - Step 1: Align image and text with a Universal Projection (UP) module.
  - Step 2: Freeze UP, train an Alignment Layer to integrate audio, video, and more.
- **Efficient and Cost-Effective:** Works well on small paired datasets, outperforming large-scale models with specialized encoders.
- **Flexible Backbone Choices:** Supports various image and text encoders (e.g., ALBERT, BERT, RoBERTa, ViT, DeiT, BeiT).

### Datasets
Update dataset paths in the config file or class `CFG`.

- **Image-Text Alignment:** [COCO Captions](https://www.kaggle.com/datasets/nikhil7280/coco-image-caption), [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), [TextCaps](https://huggingface.co/datasets/lmms-lab/TextCaps)
- **Audio Integration:** [ibriSpeech SRA](https://huggingface.co/datasets/nguyenvulebinh/asr-alignment)
- **Video Integration:** [MSR-VTT](https://huggingface.co/datasets/AlexZigma/msr-vtt)
- **X-ray Integration:** [ROCO Dataset](https://www.kaggle.com/datasets/virajbagal/roco-dataset)
- **Visual QA:** [DAQUAR](https://www.kaggle.com/datasets/tezansahu/processed-daquar-dataset)

## 📘 Usage

Train the UP for image-text alignment:
```bash
cd "text-image/addition"
python text_image.py
```

Freeze UP, train the Alignment Layer for new modalities:
```bash
mv "text-image/addition/best.pt" "audio-image/addition/text_image.pt"
cd "audio-image/addition"
python audio_image.py
```

## 🛠️ Fusion Operations
- **For Alignment:** Addition, Multiplication, Concatenation, Attention

## 🧠 Demos
- [Colab Demo](Demo)
- [Hugging Face Spaces](https://huggingface.co/spaces/bilalfaye/OneEncoder-retriever)
- Pretrained Models:
  - Text & Image → [HF Model](https://huggingface.co/bilalfaye/OneEncoder-text-image)
  - Text, Image & Audio → [HF Model](https://huggingface.co/bilalfaye/OneEncoder-text-image-audio)
  - Text, Image & Video → [HF Model](https://huggingface.co/bilalfaye/OneEncoder-text-image-video)

---
🔧 **Default Training Config:** `temperature = 2.5`, fusion via `addition`
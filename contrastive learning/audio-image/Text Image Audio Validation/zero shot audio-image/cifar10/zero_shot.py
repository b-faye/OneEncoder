"""
Author: Bilal FAYE
Date: 2023-2024
"""
################################################## PACKAGES ############################################################
################################################# PACKAGES #############################################################
# PyTorch for deep learning operations
import torch
import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms

# PyTorch data loading and utilities
# import torch.utils.data as data
# from torch.utils.data import Dataset, DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing

# Additional PyTorch modules and libraries
import os
import random
# import glob
import cv2  # OpenCV for image processing

# Transfer Learning model library
import timm

# Data manipulation and handling
# import numpy as np
# import json
import requests
import zipfile
# import collections

# COCO dataset tools
from pycocotools.coco import COCO
import numpy as np

# Hugging Face Transformers library for BERT models
from transformers import AutoProcessor, Wav2Vec2Model, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch.nn.functional as F

# Image processing and augmentations
import albumentations as A

# Visualization and progress tracking
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt

# Image handling
from PIL import Image

# Additional utility for iterating over combinations
import itertools
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import torchvision
from torch.utils.data import DataLoader
from datasets import load_dataset
import soundfile as sf
import torchaudio

from configs import CFG
from audio_image import OneEncoder as AudioImageEncoder


class CustomCIFAR10Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Convert PIL Image to NumPy array
        img = np.array(img)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']

        return img, label


def text_to_speech(text, audio_name, text_processor, audio_generator, speaker_embeddings,
                   vocoder, sample_rate=16000, device="cpu"):
    inputs = text_processor(text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        speech = audio_generator.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sf.write(audio_name, speech.cpu().numpy(), samplerate=sample_rate)

def zero_shot_classification(model, dataloader, unique_class_names, temperature=2.5):
    # Load the pretrained Wav2Vec2Model processor
    text_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    audio_generator = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(model.device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(model.device)
    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(model.device)
    audios_array = []
    for label in unique_class_names:
        text_to_speech(text=f"a {label}, this is a photo of {label}", audio_name="audio.wav",
                       text_processor=text_processor, audio_generator=audio_generator,
                       speaker_embeddings=speaker_embeddings, vocoder=vocoder)
        waveform, sample_rate = torchaudio.load("audio.wav")
        audios_array.append(waveform.squeeze().numpy())

    audios_processed = model.process_audio(audios_array)
    accuracy = {f"top_{i + 1}_accuracy": 0 for i in range(5)}
    model.eval()
    with torch.no_grad():
        audio_features = torch.mean(model.encode_audio(audios_processed), dim=1)

    with torch.no_grad():
        for id, batch in enumerate(tqdm(dataloader, desc="Testing")):
            image_tensors, labels = batch
            # Encode images
            image_features = model.text_image_encoder.encode_image(image_tensors=image_tensors)
            image_features = F.normalize(image_features, p=2, dim=-1)
            similarities = (image_features @ audio_features.T) * torch.exp(torch.tensor(temperature).to(model.device))
            audio_probs = (10.0 * similarities).softmax(dim=-1)
            _, top_k = audio_probs.cpu().topk(5, dim=-1)
            for i in range(5):
                accuracy[f"top_{i+1}_accuracy"] += torch.sum(torch.any(top_k[:, :i+1] == labels.view(-1, 1), dim=-1)).item()

    return accuracy


def main(model_path="audio_image.pt"):
    model = AudioImageEncoder(device=CFG.device)
    model = model.to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    class_names = test_set.classes
    test_set = CustomCIFAR10Dataset(test_set, transform=model.image_preprocessor)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    accuracy = zero_shot_classification(model, test_loader, class_names)
    for name, score in accuracy.items():
        print(f"{name}: {score / len(test_set.dataset)}")


if __name__ == "__main__":
    main()

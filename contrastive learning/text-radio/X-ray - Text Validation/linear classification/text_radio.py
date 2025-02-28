# PyTorch for deep learning operations
import torch
import torch.nn as nn

# PyTorch data loading and utilities
import torch.multiprocessing

# Additional PyTorch modules and libraries
import os
import random
import cv2  # OpenCV for image processing

# Transfer Learning model library
import timm

# Data manipulation and handling
import requests
import zipfile

# COCO dataset tools
from pycocotools.coco import COCO
import numpy as np

# Hugging Face Transformers library for BERT models
import torch.nn.functional as F

# Image processing and augmentations
import albumentations as A

# Visualization and progress tracking
from tqdm import tqdm
import matplotlib.pyplot as plt

# Additional utility for iterating over combinations
import itertools
from albumentations.pytorch import ToTensorV2
import pandas as pd
import math
import torchaudio
from typing import List, Dict, Any
from transformers import BertModel, BertTokenizer, AutoModel, AutoImageProcessor
import torchaudio.transforms as transforms
from datasets import load_dataset
from PIL import Image

from configs import CFG
from text_image import OneEncoder as TextImageEncoder


def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that it is indeed an image
        return True
    except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
        return False


def load_rococo_dataset():
    train_path = "../multimodal/datasets/rococo/all_data/train/radiology"
    test_path = "../multimodal/datasets/rococo/all_data/test/radiology"
    val_path = "../multimodal/datasets/rococo/all_data/validation/radiology"

    df_train = pd.read_csv(os.path.join(train_path, "traindata.csv"))
    df_test = pd.read_csv(os.path.join(test_path, "testdata.csv"))
    df_val = pd.read_csv(os.path.join(val_path, "valdata.csv"))

    train = []
    for index, row in df_train.iterrows():
        caption = row["caption"]
        radio = os.path.join(train_path, "images", row["name"])
        if os.path.exists(radio) and is_valid_image(radio):  # Check if the file exists and is valid
            train.append((caption, radio))

    test = []
    for index, row in df_test.iterrows():
        caption = row["caption"]
        radio = os.path.join(test_path, "images", row["name"])
        if os.path.exists(radio) and is_valid_image(radio):  # Check if the file exists and is valid
            test.append((caption, radio))

    val = []
    for index, row in df_val.iterrows():
        caption = row["caption"]
        radio = os.path.join(val_path, "images", row["name"])
        if os.path.exists(radio) and is_valid_image(radio):  # Check if the file exists and is valid
            val.append((caption, radio))

    return train, test, val


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, texts, radios, radio_processor, *args, **kwargs):
        super(DataGenerator, self).__init__(*args, **kwargs)
        self.texts = texts
        self.radios = radios
        self.radio_processor = radio_processor

    def __getitem__(self, index):
        item = {}

        item["text"] = self.texts[index]
        # Size : (Height, Width)
        radio = Image.open(self.radios[index])
        # Output : {"pixel_values": Size(1, 3, 518, 518)}
        radio = self.radio_processor(radio, return_tensors="pt")
        # Output : Size(3, 518, 518)
        radio = radio["pixel_values"].squeeze(0)
        item["radio"] = radio

        return item

    def __len__(self):
        return len(self.radios)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # batch is a list of dict
    # batch can be a list of tuple if item is tuple

    texts = [item["text"] for item in batch]
    radios = [item["radio"] for item in batch]
    radios = torch.stack(radios)

    return {"text": texts, "radio": radios}


def build_loaders(dataframe, radio_processor):
    dataset = DataGenerator(
        [text for text, _ in dataframe],
        [radio for _, radio in dataframe],
        radio_processor
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn
    )

    return dataloader


class AlignmentLayer(nn.Module):
    def __init__(self, input_dim=CFG.projection_dim, projection_dim=CFG.projection_dim, dropout_rate=CFG.dropout_rate, *args,
                 **kwargs):
        super(AlignmentLayer, self).__init__(*args, **kwargs)
        # Attributes
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        # Layers
        self.linear_layer1 = nn.Linear(self.input_dim, self.projection_dim)
        self.gelu = nn.GELU()
        self.linear_layer2 = nn.Linear(self.projection_dim, self.projection_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.normalization_layer = nn.LayerNorm(self.projection_dim)

    def forward(self, inputs):
        x = inputs
        x = self.linear_layer1(x)
        x = self.gelu(x)
        x = self.linear_layer2(x)
        x = self.dropout(x)
        x = self.normalization_layer(x)
        return x

    def __call__(self, inputs):
        return self.forward(inputs)


class RadioEncoder(nn.Module):
    def __init__(self, model_name=CFG.radio_name, projection_dim=CFG.projection_dim,
                 trainable=False, dropout_rate=CFG.dropout_rate, *args, **kwargs):
        super(RadioEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        # Models
        self.pretrained_encoder = AutoModel.from_pretrained(self.model_name)
        self.alignment_layer = AlignmentLayer(
            input_dim=self.pretrained_encoder.config.hidden_size,
            projection_dim=self.projection_dim,
            dropout_rate=self.dropout_rate)
        # Freeze Wav2VecModel
        for parameter in self.pretrained_encoder.parameters():
            parameter.requires_grad = self.trainable

    def forward(self, inputs):
        x = self.pretrained_encoder(inputs).last_hidden_state
        x = self.alignment_layer(x)
        return x

    def __call__(self, inputs):
        return self.forward(inputs)


class ModalityTokenEncoder(nn.Module):
    def __init__(self, projection_dim=CFG.projection_dim, token_size=CFG.token_size, device='cpu', token_dim=CFG.token_dim, *args, **kwargs):
        super(ModalityTokenEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.projection_dim = projection_dim
        self.device = device
        self.token_size = token_size
        self.token_dim = token_dim
        # Models
        audio_variance = torch.rand(1) * 0.5 + 0.1
        self.radio_token = nn.Parameter(torch.normal(mean=0, std=audio_variance.item(),
                                                      size=(self.token_size, self.token_dim)).to(self.device))

        self.token_projection = nn.Sequential(
            nn.Linear(self.token_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        )

    def forward(self):
        return self.token_projection(self.radio_token)

    def __call__(self):
        return self.forward()



class OneEncoder(nn.Module):
    def __init__(self, device='cpu', modality_token_encoder=ModalityTokenEncoder(),
                 checkpoint="text_image.pt",
                 radio_processor=AutoImageProcessor.from_pretrained("microsoft/rad-dino"),
                 sample_rate=CFG.sample_rate, radio_encoder=RadioEncoder(), *args, **kwargs):
        super(OneEncoder, self).__init__(*args, **kwargs)

        self.device = device
        self.checkpoint = checkpoint
        self.modality_token_encoder = modality_token_encoder
        self.modality_token_encoder.device = self.device
        self.text_image_encoder = TextImageEncoder(device=self.device)
        self.text_image_encoder.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        self.radio_processor = radio_processor
        self.sample_rate = sample_rate
        self.radio_encoder = radio_encoder
        self.temperature = nn.Parameter(torch.tensor(0.07).to(self.device))

        # Freeze
        for parameter in self.text_image_encoder.parameters():
            parameter.requires_grad = False

    def encode_radio(self, pil_radios=None, radios=None):
        """
        :param pil_radios: list of pillow images
        :param radios: preprocessed image
        :return: tensor
        """
        if pil_radios is not None:
            tensors = self.radio_processor(pil_radios, return_tensors="pt")["pixel_values"].to(self.device)
        else:
            tensors = radios.to(self.device)
        features = self.radio_encoder(tensors)
        radio_token = self.modality_token_encoder()
        outputs = self.text_image_encoder.universal_projection_encoder([features, radio_token]).last_hidden_state
        return outputs



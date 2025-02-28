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

    def forward(self, inputs):
        texts = inputs["text"]
        radios = inputs["radio"]

        text_features = self.text_image_encoder.encode_text(texts=texts)
        radio_features = torch.mean(self.encode_radio(radios=radios), dim=1)

        # L2 normalization
        text_features = F.normalize(text_features, p=2, dim=-1)
        radio_features = F.normalize(radio_features, p=2, dim=-1)

        t = torch.clamp(self.temperature.data, min=torch.tensor(2.5).to(self.device),
                        max=torch.tensor(3.0).to(self.device))

        # Compute loss
        logits = (text_features @ radio_features.T) * torch.exp(t)
        labels = torch.arange(logits.size(0)).to(self.device)
        loss_i = F.cross_entropy(logits, labels, reduction='mean')
        loss_t = F.cross_entropy(logits.t(), labels, reduction='mean')
        loss = (loss_i + loss_t) / 2.0

        return loss

    def __call__(self, inputs):
        return self.forward(inputs)


################################################### TRAINING STEP ######################################################
################################################# TRAINING STEP ########################################################

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        # val: mean loss of the batch
        # count: Total of samples (sum of samples in all mini-batch)
        # sum: sum of loss (*count allow to pass to mean to sum)
        # avg: mean loss of all batches
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    # List of dictionary, each dictionary is a batch
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()
        count = len(batch["text"])
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        loss = model(batch)

        count = len(batch["text"])
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


################################################### TRAINING STEP ######################################################
################################################# TRAINING STEP ########################################################

def main(basic_train=True):
    training_pairs, test_pairs, validation_pairs = load_rococo_dataset()
    training_pairs = training_pairs + test_pairs
    random.shuffle(training_pairs)
    random.shuffle(validation_pairs)

    print("Number of training images: {}".format(len(training_pairs)))
    print("Number of validation images: {}".format(len(validation_pairs)))
    radio_processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")

    train_loader = build_loaders(training_pairs, radio_processor)
    val_loader = build_loaders(validation_pairs, radio_processor)

    # Create the training model
    model = OneEncoder(device=CFG.device)
    model = model.to(CFG.device)

    if basic_train:
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=CFG.weight_decay, lr=CFG.lr)
    else:
        parameters = [
            {"params": model.text_image_encoder.parameters(), "lr": CFG.radio_encoder_lr},
            {"params": model.radio_encoder.pretrained_encoder.parameters(), "lr": CFG.radio_encoder_lr},
            {
                "params": itertools.chain(
                    model.modality_token_encoder.parameters(), model.radio_encoder.alignment_layer.parameters(),
                ),
                "lr": CFG.lr, "weight_decay": CFG.weight_decay
            },
            {"params": [model.temperature], "lr": CFG.lr, "weight_decay": CFG.weight_decay}
        ]
        optimizer = torch.optim.AdamW(parameters, weight_decay=0.)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

    num_bad_epochs = 0
    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(model.temperature.data)
        print("Epoch: %d" % (epoch + 1))
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, "epoch")
        print(f"Epoch: {epoch + 1}, train loss: {train_loss}")
        model.eval()
        with torch.no_grad():
            val_loss = valid_epoch(model, val_loader)
            print(f"Epoch: {epoch + 1}, val loss: {val_loss}")
        if val_loss.avg < best_loss:
            best_loss = val_loss.avg
            num_bad_epochs = 0
            torch.save(model.state_dict(), "best.pt")
            print("Saved best model!")
        else:
            if epoch >= CFG.patience - 1:
                num_bad_epochs += 1
            if num_bad_epochs >= CFG.patience:
                print(f"Early stopping at epoch {epoch + 1}. Restoring best weights...")
                break
        lr_scheduler.step(val_loss.avg)
    torch.save(model.state_dict(), "last.pt")
    # Free GPU
    model = None
    optimizer = None
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

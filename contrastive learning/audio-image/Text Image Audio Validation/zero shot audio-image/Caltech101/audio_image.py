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
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertConfig, DistilBertTokenizer
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
from transformers import AutoProcessor, Wav2Vec2Model
import torchaudio.transforms as transforms
from IPython.display import Audio



from configs import CFG
from text_image import OneEncoder as TextImageEncoder

def make_pairs(annotation_json_file, image_dir, audio_path, max_captions=3):
    coco = COCO(annotation_json_file)
    image_ids = list(coco.imgs.keys())

    image_audio = []
    for image_id in image_ids:
        num_caption = 0
        # image_info: list of one dict [{}]
        image_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(image_dir, image_info["file_name"])
        # annotation_ids: list of caption ids to image_id []
        annotation_ids = coco.getAnnIds(image_id)
        # annotations: list of dictionary, each dictionary for one caption [{}, {}, ..., {}]
        annotations = coco.loadAnns(annotation_ids)
        for annotation in annotations:
            if 'caption' in list(annotation.keys()) and num_caption < max_captions:
                image_audio.append((image_path, os.path.join(audio_path, f"speech_{annotation['id']}.wav")))
                num_caption += 1

    return image_audio


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, images, audios, transformer, audio_processor, sample_rate=CFG.sample_rate, *args, **kwargs):
        super(DataGenerator, self).__init__(*args, **kwargs)
        # Image processor
        self.transformer = transformer
        # List of images
        self.images = images
        # List of captions
        self.audios = audios
        # audio
        self.audio_processor = audio_processor
        self.sample_rate = sample_rate

    def __getitem__(self, index):
        # Select caption in position index
        item = {}

        # image
        image = cv2.imread(f"{self.images[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transformer(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()

        # convert caption to audio
        waveform, sample_rate = torchaudio.load(self.audios[index])
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        item["audio"] = waveform
        return item

    def __len__(self):
        return len(self.audios)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Separate images and audio from batch
    images = [item["image"] for item in batch]
    audios = [item["audio"].squeeze().numpy() for item in batch]

    # Stack images (they should all be the same size)
    images = torch.stack(images)

    # Return audio data as a list (no padding)
    return {"image": images, "audio": audios}

def image_processor(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.image_size, CFG.image_size, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.image_size, CFG.image_size, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
            ]
        )


def build_loaders(dataframe, mode, audio_processor, sample_rate=CFG.sample_rate):
    transformer = image_processor(mode=mode)
    dataset = DataGenerator(
        [img for img, _ in dataframe],
        [audio for _, audio in dataframe],
        transformer=transformer,
        audio_processor=audio_processor,
        sample_rate=sample_rate
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        shuffle=True if mode == "train" else False,
    )

    return dataloader


class AlignmentLayer(nn.Module):
    def __init__(self, input_dim=CFG.projection_dim, projection_dim=CFG.projection_dim, dropout_rate=CFG.dropout_rate, *args, **kwargs):

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

class AudioEncoder(nn.Module):
    def __init__(self, model_name=CFG.audio_name, projection_dim=CFG.projection_dim,
                 trainable=False, dropout_rate=CFG.dropout_rate, *args, **kwargs):

        super(AudioEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        # Models
        self.pretrained_encoder = Wav2Vec2Model.from_pretrained(self.model_name)
        self.alignment_layer = AlignmentLayer(
                                              input_dim=self.pretrained_encoder.config.hidden_size,
                                              projection_dim=self.projection_dim,
                                              dropout_rate=self.dropout_rate)
        # Freeze Wav2VecModel
        for parameter in self.pretrained_encoder.parameters():
            parameter.requires_grad = self.trainable
        # Unfreeze not initialized layers
        newly_initialized_layers = [
            'encoder.pos_conv_embed.conv.parametrizations.weight.original0',
            'encoder.pos_conv_embed.conv.parametrizations.weight.original1',
            'masked_spec_embed'
        ]
        for name, param in self.pretrained_encoder.named_parameters():
            if any(layer_name in name for layer_name in newly_initialized_layers):
                param.requires_grad = True

    def forward(self, inputs):

        x = self.pretrained_encoder(inputs['input_values']).last_hidden_state
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
        self.audio_token = nn.Parameter(torch.normal(mean=0, std=audio_variance.item(),
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
        return self.token_projection(self.audio_token)

    def __call__(self):
        return self.forward()

class OneEncoder(nn.Module):
    def __init__(self, device='cpu', modality_token_encoder=ModalityTokenEncoder(), checkpoint="text_image.pt",
                 audio_processor=AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h"),
                 sample_rate=CFG.sample_rate, audio_encoder=AudioEncoder(), *args, **kwargs):
        super(OneEncoder, self).__init__(*args, **kwargs)

        self.device = device
        self.checkpoint = checkpoint
        self.modality_token_encoder = modality_token_encoder
        self.modality_token_encoder.device = self.device
        self.text_image_encoder = TextImageEncoder(device=self.device)
        self.text_image_encoder.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        self.audio_processor = audio_processor
        self.sample_rate = sample_rate
        self.audio_encoder = audio_encoder
        self.temperature = nn.Parameter(torch.tensor(0.07).to(self.device))

        # Freeze
        for parameter in self.text_image_encoder.parameters():
            parameter.requires_grad = False

    def load_audio(self, audio_path):
        waveform, original_sample_rate = torchaudio.load(audio_path)
        # If the audio needs to be resampled
        if original_sample_rate != self.sample_rate:
            resampler = transforms.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # mono sound -> output shape: torch.Size(1, dim)
        # Stereo sound -> output shape: torch.Size(2, dim)
        # Surround sound -> output shape: torch.Size(n, dim)
        return waveform

    def process_audio(self, audios):
        # audios: list of numpy array
        x = self.audio_processor(audios, sampling_rate=self.sample_rate, return_tensors="pt", padding=True, max_length=15*self.sample_rate, truncation=True)
        #x = self.audio_processor(audios, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        return x

    def encode_audio(self, audios):
        # audios: torch 2D (batch, dim)
        audio_embeddings = self.audio_encoder(audios.to(self.device))
        modality_token = self.modality_token_encoder()
        audio_features = self.text_image_encoder.universal_projection_encoder([audio_embeddings, modality_token]).last_hidden_state
        return audio_features.float()

    def matching_image_audio(self, audios, image_paths=None, image_tensors=None,
                              normalize=True, top_k=None, strategy="similarity", temperature=0.0):
        # audios is of shape {"input_values":torch.Size([N, dim])}
        wav_features = torch.mean(self.encode_audio(audios), dim=1)
        image_features = self.text_image_encoder.encode_image(image_paths=image_paths, image_tensors=image_tensors)
        if normalize:
            image_features = F.normalize(image_features, p=2, dim=-1)
            wav_features = F.normalize(wav_features, p=2, dim=-1)
        dot_similarities = (image_features @ wav_features.T) * torch.exp(torch.tensor(temperature).to(self.device))
        if strategy == 'softmax':
            dot_similarities = (float(audios["input_values"].shape[0]) * dot_similarities).softmax(dim=-1)
        if top_k is not None:
            top_probs, top_labels = dot_similarities.cpu().topk(top_k, dim=-1)
            return top_probs, top_labels
        else:
            return dot_similarities, None

    def matching_text_audio(self, audios, texts, normalize=True, top_k=None, strategy="similarity", temperature=0.0):
        # audios is of shape {"input_values":torch.Size([N, dim])}
        wav_features = torch.mean(self.encode_audio(audios), dim=1)
        text_features = self.text_image_encoder.encode_text(texts=texts)
        if normalize:
            text_features = F.normalize(text_features, p=2, dim=-1)
            wav_features = F.normalize(wav_features, p=2, dim=-1)
        dot_similarities = (text_features @ wav_features.T) * torch.exp(torch.tensor(temperature).to(self.device))
        if strategy == 'softmax':
            dot_similarities = (float(audios["input_values"].shape[0]) * dot_similarities).softmax(dim=-1)
        if top_k is not None:
            top_probs, top_labels = dot_similarities.cpu().topk(top_k, dim=-1)
            return top_probs, top_labels
        else:
            return dot_similarities, None

    def image_retrieval(self, query, image_paths, image_embeddings=None, temperature=0.0, n=9, plot=False, display_audio=False):
        # query is of shape {"input_values":torch.Size([1, dim])}
        wav_embeddings = torch.mean(self.encode_audio(audios=query), dim=1)
        if image_embeddings is None:
            image_embeddings = self.text_image_encoder.encode_image(image_paths=image_paths)

        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        wav_embeddings_n = F.normalize(wav_embeddings, p=2, dim=-1)
        dot_similarity = (wav_embeddings_n @ image_embeddings_n.T) * torch.exp(
            torch.tensor(temperature).to(self.device))
        if n > len(image_paths):
            n = len(image_paths)
        values, indices = torch.topk(dot_similarity.cpu().squeeze(0), n)
        if plot:
            nrows = int(np.sqrt(n))
            ncols = int(np.ceil(n / nrows))
            matches = [image_paths[idx] for idx in indices]
            fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
            for match, ax in zip(matches, axes.flatten()):
                image = self.text_image_encoder.load_image(f"{match}")
                ax.imshow(image)
                ax.axis("off")
            plt.savefig("img.png")
            if display_audio:
                fig.suptitle(display(Audio(query['input_values'], rate=self.sample_rate)))
            plt.show()
        return values, indices




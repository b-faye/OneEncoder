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
from datasets import load_dataset
from IPython.display import Audio

from configs import CFG
from text_image import OneEncoder as TextImageEncoder


def load_asr_dataset():
    dataset = load_dataset("nguyenvulebinh/asr-alignment", "libris", trust_remote_code=True)
    """
    OUTPUT
    DatasetDict({
        train: Dataset({
            features: ['id', 'text', 'audio', 'words', 'word_start', 'word_end', 'entity_start', 'entity_end', 'entity_label'],
            num_rows: 281241
        })
        valid: Dataset({
            features: ['id', 'text', 'audio', 'words', 'word_start', 'word_end', 'entity_start', 'entity_end', 'entity_label'],
            num_rows: 5559
        })
    })
    """
    # Check text-audio
    train = [(example['text'], example['audio']['array']) for example in dataset['train']]
    valid = [(example['text'], example['audio']['array']) for example in dataset['valid']]

    return train, valid


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, datasets, *args, **kwargs):
        super(DataGenerator, self).__init__(*args, **kwargs)
        self.datasets = datasets

    def __getitem__(self, index):
        item = {}
        item["text"] = self.datasets[index]['text']
        item["audio"] = self.datasets[index]['audio']['array']
        return item

    def __len__(self):
        return len(self.datasets)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # batch is a list of dict
    # batch can be a list of tuple if item is tuple

    texts = [item["text"] for item in batch]
    audios = [item["audio"] for item in batch]

    # Return audio data as a list (no padding)
    return {"text": texts, "audio": audios}


def build_loaders(dataframe):
    dataset = DataGenerator(
        dataframe
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn
    )

    return dataloader


class AlignmentLayer(nn.Module):
    def __init__(self, input_dim=CFG.projection_dim, projection_dim=CFG.projection_dim, dropout_rate=CFG.dropout_rate,
                 *args, **kwargs):
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

        x = self.pretrained_encoder(inputs['input_values'].float()).last_hidden_state
        x = self.alignment_layer(x)
        return x

    def __call__(self, inputs):
        return self.forward(inputs)


class ModalityTokenEncoder(nn.Module):
    def __init__(self, projection_dim=CFG.projection_dim, token_size=CFG.token_size, device='cpu',
                 token_dim=CFG.token_dim, *args, **kwargs):
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
    def __init__(self, device='cpu', modality_token_encoder=ModalityTokenEncoder(),
                 checkpoint="../text_image_add/text_image.pt",
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
        x = self.audio_processor(audios, sampling_rate=self.sample_rate, return_tensors="pt", padding=True,
                                 max_length=20 * self.sample_rate, truncation=True)
        # x = self.audio_processor(audios, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        return x

    def encode_audio(self, audios):
        # audios: torch 2D (batch, dim)
        audio_embeddings = self.audio_encoder(audios.to(self.device))
        modality_token = self.modality_token_encoder()
        audio_features = self.text_image_encoder.universal_projection_encoder(
            [audio_embeddings, modality_token]).last_hidden_state
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

    def image_retrieval(self, query, image_paths, image_embeddings=None, temperature=0.0, n=9, plot=False,
                        display_audio=False):
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

    def forward(self, inputs):
        texts = inputs["text"]
        audios = inputs["audio"]

        audios = self.audio_processor(audios, sampling_rate=self.sample_rate, return_tensors="pt", padding=True,
                                      max_length=20 * self.sample_rate, truncation=True)

        text_features = self.text_image_encoder.encode_text(texts=texts)
        audio_features = torch.mean(self.encode_audio(audios), dim=1)

        # L2 normalization
        text_features = F.normalize(text_features, p=2, dim=-1)
        audio_features = F.normalize(audio_features, p=2, dim=-1)

        t = torch.clamp(self.temperature.data, min=torch.tensor(2.5).to(self.device),
                        max=torch.tensor(3.0).to(self.device))

        # Compute loss
        logits = (text_features @ audio_features.T) * torch.exp(t)
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
    # training_pairs, validation_pairs = load_asr_dataset()
    # random.shuffle(training_pairs)
    # random.shuffle(validation_pairs)

    # print("Number of training images: {}".format(len(training_pairs)))
    # print("Number of validation images: {}".format(len(validation_pairs)))
    dataset = load_dataset("nguyenvulebinh/asr-alignment", "libris", trust_remote_code=True)

    train_loader = build_loaders(dataset['train'])
    val_loader = build_loaders(dataset['valid'])

    # Create the training model
    model = OneEncoder(device=CFG.device)
    model = model.to(CFG.device)

    if basic_train:
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=CFG.weight_decay, lr=CFG.lr)
    else:
        parameters = [
            {"params": model.text_image_encoder.parameters(), "lr": CFG.audio_encoder_lr},
            {"params": model.audio_encoder.pretrained_encoder.parameters(), "lr": CFG.audio_encoder_lr},
            {
                "params": itertools.chain(
                    model.modality_token_encoder.parameters(), model.audio_encoder.alignment_layer.parameters(),
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



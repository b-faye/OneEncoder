################################################## PACKAGES ############################################################
################################################# PACKAGES #############################################################
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
from transformers import BertModel, BertTokenizer, AutoImageProcessor, VideoMAEModel
import torch.nn.functional as F

# Image processing and augmentations
import albumentations as A

# Visualization and progress tracking
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
import av # pip install av

# Additional utility for iterating over combinations
import itertools
from albumentations.pytorch import ToTensorV2
import pandas as pd
import math
from configs import CFG
from text_image import TextImageEncoder


def load_msr_vtt_dataset():
    # Load the MSR-VTT dataset
    dataset = load_dataset('AlexZigma/msr-vtt')
    """
    DatasetDict({
    train: Dataset({
        features: ['video_id', 'caption', 'sen_id', 'category', 'url', 'start time', 'end time', 'split', 'id', '__index_level_0__'],
        num_rows: 6513
    })
    val: Dataset({
        features: ['video_id', 'caption', 'sen_id', 'category', 'url', 'start time', 'end time', 'split', 'id', '__index_level_0__'],
        num_rows: 497
    })
    })
    """
    # Access the train and test splits
    train_dataset = dataset['train']
    val_dataset = dataset['val']

    train = []
    for item in train_dataset:
        image_path = os.path.join("../../datasets/msr-vtt/TrainValVideo", f"{item['video_id']}.mp4")
        train.append((item['caption'], image_path))

    val = []
    for item in val_dataset:
        image_path = os.path.join("../../datasets/msr-vtt/TrainValVideo", f"{item['video_id']}.mp4")
        val.append((item['caption'], image_path))

    return train, val


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    """
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, videos, video_processor):
        self.texts = texts
        self.videos = videos
        self.video_processor = video_processor

    def __getitem__(self, index):
        item = {}
        item['text'] = self.texts[index]

        # Load the video using a context manager to ensure it is closed properly
        with av.open(self.videos[index]) as container:
            # Sample 16 frames
            indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            # Read the video (16, Height, Width, 3)
            video = read_video_pyav(container, indices)
            # Process video (1, 16, 3, 224, 224)
            video = self.video_processor(list(video), return_tensors='pt')
            item['video'] = video['pixel_values']

        return item

    def __len__(self):
        return len(self.texts)


def collate_fn(batch):

    # batch = list of dicts
    texts = [item['text'] for item in batch]
    videos = [item['video'] for item in batch]
    # (batch_size, 16, 3, 224, 224)
    videos = torch.cat(videos)

    return {'text': texts, 'video': videos}


def build_loaders(dataframe, video_processor):
    texts = [text for text, _ in dataframe]
    videos = [video for _, video in dataframe]
    dataset = Dataset(texts, videos, video_processor)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        collate_fn=collate_fn
    )

    return dataloader


class AlignmentLayer(nn.Module):
    def __init__(self, input_dim=768, projection_dim=CFG.projection_dim, dropout_rate=CFG.dropout_rate, *args, **kwargs):

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


class VideoEncoder(nn.Module):
    def __init__(self, model_name=CFG.video_name, projection_dim=CFG.projection_dim,
                 trainable=False, dropout_rate=CFG.dropout_rate, *args, **kwargs):

        super(VideoEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        # Models
        self.pretrained_encoder = VideoMAEModel.from_pretrained(self.model_name)
        self.alignment_layer = AlignmentLayer(
                                              input_dim=self.pretrained_encoder.config.hidden_size,
                                              projection_dim=self.projection_dim,
                                              dropout_rate=self.dropout_rate)
        # Freeze VideoMAE
        for parameter in self.pretrained_encoder.parameters():
            parameter.requires_grad = self.trainable

    def forward(self, inputs):

        x = self.pretrained_encoder(inputs).last_hidden_state
        x = self.alignment_layer(x)
        return x

    def __call__(self, inputs):
        return self.forward(inputs)


class ModalityTokenEncoder(nn.Module):
    def __init__(self, projection_dim=CFG.projection_dim, token_size=CFG.token_size, device='cpu', *args, **kwargs):
        super(ModalityTokenEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.projection_dim = projection_dim
        self.device = device
        self.token_size = token_size
        # Models
        video_variance = torch.rand(1) * 0.5 + 0.1
        self.video_token = nn.Parameter(torch.normal(mean=0, std=video_variance.item(),
                                                      size=(self.token_size, self.projection_dim)).to(self.device))

    def forward(self):
        return self.video_token

    def __call__(self):
        return self.forward()


class TextVideoEncoder(nn.Module):
    def __init__(self, device='cpu', modality_token_encoder=ModalityTokenEncoder(), checkpoint="../text_image_add/text_image.pt",
                 video_processor=AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base"),
                 video_encoder=VideoEncoder(), *args, **kwargs):
        super(TextVideoEncoder, self).__init__(*args, **kwargs)

        self.device = device
        self.checkpoint = checkpoint
        self.modality_token_encoder = modality_token_encoder
        self.modality_token_encoder.device = self.device
        self.text_image_encoder = TextImageEncoder(device=self.device)
        self.text_image_encoder.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        self.video_processor = video_processor
        self.video_encoder = video_encoder
        self.temperature = nn.Parameter(torch.tensor(0.07).to(self.device))

        # Freeze
        for parameter in self.text_image_encoder.parameters():
            parameter.requires_grad = False

    @classmethod
    def load_video(cls, video_path):
        container = av.open(video_path)
        return container

    @classmethod
    def read_video_pyav(cls, container, indices):
        """
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        """
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    @classmethod
    def sample_frame_indices(cls, clip_len, frame_sample_rate, seg_len):
        """
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        """
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    def encode_video(self, videos):
        """
        :param videos: torch.Size([batch, 16, 3, 224, 224])
        :return: torch.Size([batch, 1568, 768])
        """
        video_features = self.video_encoder(videos.to(self.device))
        modality_token_features = self.modality_token_encoder()
        outputs = self.text_image_encoder.universal_projection_encoder([video_features, modality_token_features]).last_hidden_state

        return outputs

    def forward(self, inputs):
        texts = inputs["text"]
        videos = inputs["video"]

        text_features = self.text_image_encoder.encode_text(texts=texts)
        video_features = torch.mean(self.encode_video(videos=videos), dim=1)

        # L2 normalization
        text_features = F.normalize(text_features, p=2, dim=-1)
        video_features = F.normalize(video_features, p=2, dim=-1)

        t = torch.clamp(self.temperature.data, min=torch.tensor(2.5).to(self.device),
                        max=torch.tensor(3.0).to(self.device))

        # Compute loss
        logits = (text_features @ video_features.T) * torch.exp(t)
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

    training_pairs, validation_pairs = load_msr_vtt_dataset()
    random.shuffle(training_pairs)
    random.shuffle(validation_pairs)

    print("Number of training images: {}".format(len(training_pairs)))
    print("Number of validation images: {}".format(len(validation_pairs)))
    video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    train_loader = build_loaders(training_pairs, video_processor)
    val_loader = build_loaders(validation_pairs, video_processor)

    # Create the training model
    model = TextVideoEncoder(device=CFG.device)
    model = model.to(CFG.device)

    if basic_train:
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=CFG.weight_decay, lr=CFG.lr)
    else:
        parameters = [
            {"params": model.text_image_encoder.parameters(), "lr": CFG.video_encoder_lr},
            {"params": model.video_encoder.pretrained_encoder.parameters(), "lr": CFG.video_encoder_lr},
            {
                "params": itertools.chain(
                    model.modality_token_encoder.parameters(), model.video_encoder.alignment_layer.parameters(),
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
        print("Epoch: %d" % (epoch+1))
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, "epoch")
        print(f"Epoch: {epoch+1}, train loss: {train_loss}")
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

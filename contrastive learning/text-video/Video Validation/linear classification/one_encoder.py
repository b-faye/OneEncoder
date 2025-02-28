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
from text_video import OneEncoder as TextVideoEncoder

from huggingface_hub import hf_hub_download
import tarfile


def load_dataset(path, label2id):
    dataset = []
    for dir in os.listdir(path):
        for file in os.listdir(os.path.join(path, dir)):
            if file.endswith(".avi"):
                dataset.append((
                    os.path.join(path, dir, file),
                    label2id[dir]
                ))
    return dataset


def load_ucf101_dataset():
    hf_dataset_identifier = "sayakpaul/ucf101-subset"
    filename = "UCF101_subset.tar.gz"
    file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")

    with tarfile.open(file_path) as t:
        t.extractall(".")

    label2id = {'ApplyEyeMakeup': 0,
                'ApplyLipstick': 1,
                'Archery': 2,
                'BabyCrawling': 3,
                'BalanceBeam': 4,
                'BandMarching': 5,
                'BaseballPitch': 6,
                'Basketball': 7,
                'BasketballDunk': 8,
                'BenchPress': 9
    }

    train = load_dataset("UCF101_subset/train", label2id)
    test = load_dataset("UCF101_subset/test", label2id)
    val = load_dataset("UCF101_subset/val", label2id)

    return train, test, val


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


class CustomUCFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        video_path = self.dataset[idx][0]
        label = int(self.dataset[idx][1])

        # Load the video
        container = av.open(video_path)
        # sample 16 frames
        indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        # read the video ((16, Height, Width, 3)
        video = read_video_pyav(container, indices)
        # process video (1, 16, 3, 224, 224)
        video = self.video_processor(list(video), return_tensors='pt')['pixel_values']

        return video, label


def collate_fn(batch):
    videos, labels = zip(*batch)
    videos = torch.cat(videos)
    labels = torch.tensor(labels)

    return videos, labels


class LinearClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes=10, trainable=False, device='cpu', *args, **kwargs):
        super(LinearClassifier, self).__init__(*args, **kwargs)
        self.model = pretrained_model
        self.num_classes = num_classes
        self.trainable = trainable
        self.device = device
        self.classifier = nn.Sequential(
            nn.Linear(self.model.text_image_encoder.universal_projection_encoder.input_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),

            nn.Linear(32, num_classes)
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = self.trainable
        self.to(self.device)
        self.model.to(self.device)
        self.classifier.to(self.device)

    def forward(self, inputs):
        radio_features = self.model.encode_video(videos=inputs)
        logits = self.classifier(radio_features)
        return logits

    def accuracy(self, data_loader):
        top_accuracy = {f"top_{i+1}_accuracy": 0 for i in range(5)}
        total_samples = 0
        with torch.no_grad():
            self.eval()
            for inputs, labels in tqdm(data_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                total_samples += labels.size(0)
                logits = self(inputs)
                _, predicted_top_k = torch.topk(logits, 5, dim=1)
                for i in range(5):
                    top_accuracy[f"top_{i+1}_accuracy"] += torch.sum(torch.any(
                        predicted_top_k[:, :i+1] == labels.view(-1, 1), dim=-1)).item()

        for name in top_accuracy:
            top_accuracy[name] /= total_samples

        return top_accuracy

    def __call__(self, inputs):
        return self.forward(inputs)


def main(model_path="text_video.pt", trainable=False, epochs=100, device=CFG.device):
    pretrained_model = TextVideoEncoder(device=device)
    pretrained_model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    train, test, val = load_ucf101_dataset()

    train_data = CustomUCFDataset(train, pretrained_model.video_processor)
    test_data = CustomUCFDataset(test, pretrained_model.video_processor)
    val_data = CustomUCFDataset(val, pretrained_model.video_processor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2,
                                             collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=2,
                                            collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True, num_workers=2,
                                              collate_fn=collate_fn)

    model = LinearClassifier(pretrained_model=pretrained_model, device=device, trainable=trainable)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        if (epoch + 1) % 10 == 0:
            # Validate the model on the test set
            top_accuracy = model.accuracy(test_loader)
            print(top_accuracy)

    # Save the fine-tuned model if needed
    torch.save(model.state_dict(), 'one_encoder_linear_classifier.pt')


if __name__ == "__main__":
    main()





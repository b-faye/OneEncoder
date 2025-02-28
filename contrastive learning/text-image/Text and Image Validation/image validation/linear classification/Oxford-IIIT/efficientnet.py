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
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertConfig, DistilBertTokenizer
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
from configs import CFG
from text_image import OneEncoder as TextImageEncoder


class CustomOxfordIIITDataset(Dataset):
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


class LinearClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes=37, trainable=False, device='cpu', *args, **kwargs):
        super(LinearClassifier, self).__init__(*args, **kwargs)
        self.model = pretrained_model
        self.num_classes = num_classes
        self.trainable = trainable
        self.device = device
        self.classifier = nn.Sequential(
            # self.pretrained_model.fc.in_features = 1280
            nn.Linear(1280, 512),
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

            nn.Linear(64, num_classes)
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = self.trainable
        self.to(self.device)
        self.model.to(self.device)
        self.classifier.to(self.device)

    def forward(self, inputs):
        image_features = self.model.encode_image(image_tensors=inputs)
        logits = self.classifier(image_features)
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


def main(trainable=False, epochs=100, device=CFG.device):
    text_image_encoder = TextImageEncoder(device=device)
    pretrained_model = torchvision.models.efficientnet_b0(pretrained=True)
    pretrained_model.classifier[1] = nn.Identity()  # Remove the final fully connected layer
    # Load Data
    train_data = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval', download=True)
    train_data = CustomOxfordIIITDataset(train_data, transform=text_image_encoder.image_preprocessor)
    test_data = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True)
    test_data = CustomOxfordIIITDataset(test_data, transform=text_image_encoder.image_preprocessor)
    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)
    # Build model
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
    torch.save(model.state_dict(), 'fine_tuned_model.pt')


if __name__ == "__main__":
    main()

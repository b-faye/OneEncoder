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


def zero_shot_classification(model, dataloader, unique_class_names, temperature=2.5):

    accuracy = {f"top_{i+1}_accuracy": 0 for i in range(5)}
    # Encode label names
    text_features = model.encode_text(texts=[f"a photo of a {name}" for name in unique_class_names])
    text_features = F.normalize(text_features, p=2, dim=-1)
    model.eval()
    with torch.no_grad():
        for id, batch in enumerate(tqdm(dataloader, desc="Testing")):
            image_tensors, labels = batch
            # Encode images
            image_features = model.encode_image(image_tensors=image_tensors)
            image_features = F.normalize(image_features, p=2, dim=-1)
            similarities = (image_features @ text_features.T) * torch.exp(torch.tensor(temperature).to(model.device))
            text_probs = (37.0 * similarities).softmax(dim=-1)
            _, top_k = text_probs.cpu().topk(5, dim=-1)
            for i in range(5):
                accuracy[f"top_{i+1}_accuracy"] += torch.sum(torch.any(top_k[:, :i+1] == labels.view(-1, 1), dim=-1)).item()

    return accuracy


def main(model_path="text_image.pt"):
    model = TextImageEncoder(device=CFG.device)
    model = model.to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    test_set = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True)
    class_names = test_set.classes
    test_set = CustomOxfordIIITDataset(test_set, transform=model.image_preprocessor)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    accuracy = zero_shot_classification(model, test_loader, class_names)
    for name, score in accuracy.items():
        print(f"{name}: {score / len(test_set.dataset)}")


if __name__ == "__main__":
    main()

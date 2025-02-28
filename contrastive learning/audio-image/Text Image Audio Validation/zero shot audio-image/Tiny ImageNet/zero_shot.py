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
import urllib.request

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


def download_and_unzip_tiny_imagenet(download_dir='../datasets/tiny_imagenet'):
    # Create the download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Download TinyImageNet dataset
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_file_path = os.path.join(download_dir, 'tiny_imagenet.zip')
    urllib.request.urlretrieve(url, zip_file_path)

    # Extract the downloaded zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)

    # Remove the zip file
    os.remove(zip_file_path)

    print("TinyImageNet dataset downloaded and unzipped successfully!")


class CustomTinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.classes = os.listdir(os.path.join(root_dir, 'train'))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        if self.train:
            self.data_dir = os.path.join(root_dir, 'train')
            self.annotations_file = os.path.join(root_dir, 'wnids.txt')
            with open(self.annotations_file, 'r') as f:
                self.classes = sorted([line.strip() for line in f.readlines()])
            self.image_paths = []
            for cls in self.classes:
                images_dir = os.path.join(self.data_dir, cls, 'images')
                for img_name in os.listdir(images_dir):
                    self.image_paths.append((os.path.join(images_dir, img_name), self.class_to_idx[cls]))
        else:
            self.data_dir = os.path.join(root_dir, 'val')
            self.annotations_file = os.path.join(root_dir, 'val', 'val_annotations.txt')
            self.image_paths = []
            with open(self.annotations_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    img_cls = parts[1]
                    img_cls_idx = self.class_to_idx[img_cls]
                    img_path = os.path.join(self.data_dir, 'images', img_name)
                    self.image_paths.append((img_path, img_cls_idx))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        if self.transform:
            img = self.transform(image=img)['image']

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
            audio_probs = (200.0 * similarities).softmax(dim=-1)
            _, top_k = audio_probs.cpu().topk(5, dim=-1)
            for i in range(5):
                accuracy[f"top_{i+1}_accuracy"] += torch.sum(torch.any(top_k[:, :i+1] == labels.view(-1, 1), dim=-1)).item()

    return accuracy


def main(model_path="audio_image.pt"):
    model = AudioImageEncoder(device=CFG.device)
    model = model.to(CFG.device)
    test_set = CustomTinyImageNetDataset("../datasets/tiny_imagenet/tiny-imagenet-200",
                                         model.image_preprocessor, train=False)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    class_names = [
        "tench",
        "goldfish",
        "great white shark",
        "tiger shark",
        "hammerhead",
        "electric ray",
        "stingray",
        "cock",
        "hen",
        "ostrich",
        "brambling",
        "goldfinch",
        "house finch",
        "junco",
        "indigo bunting",
        "robin",
        "bulbul",
        "jay",
        "magpie",
        "chickadee",
        "water ouzel",
        "kite",
        "bald eagle",
        "vulture",
        "great grey owl",
        "European fire salamander",
        "common newt",
        "eft",
        "spotted salamander",
        "axolotl",
        "bullfrog",
        "tree frog",
        "tailed frog",
        "loggerhead",
        "leatherback turtle",
        "mud turtle",
        "terrapin",
        "box turtle",
        "banded gecko",
        "common iguana",
        "American chameleon",
        "whiptail",
        "agama",
        "frilled lizard",
        "alligator lizard",
        "Gila monster",
        "green lizard",
        "African chameleon",
        "Komodo dragon",
        "African crocodile",
        "American alligator",
        "triceratops",
        "thunder snake",
        "ringneck snake",
        "hognose snake",
        "green snake",
        "king snake",
        "garter snake",
        "water snake",
        "vine snake",
        "night snake",
        "boa constrictor",
        "rock python",
        "Indian cobra",
        "green mamba",
        "sea snake",
        "horned viper",
        "diamondback",
        "sidewinder",
        "trilobite",
        "harvestman",
        "scorpion",
        "black and gold garden spider",
        "barn spider",
        "garden spider",
        "black widow",
        "tarantula",
        "wolf spider",
        "tick",
        "centipede",
        "black grouse",
        "ptarmigan",
        "ruffed grouse",
        "prairie chicken",
        "peacock",
        "quail",
        "partridge",
        "African grey",
        "macaw",
        "sulphur-crested cockatoo",
        "lorikeet",
        "coucal",
        "bee eater",
        "hornbill",
        "hummingbird",
        "jacamar",
        "toucan",
        "drake",
        "red-breasted merganser",
        "goose",
        "black swan",
        "tusker",
        "echidna",
        "platypus",
        "wallaby",
        "koala",
        "wombat",
        "jellyfish",
        "sea anemone",
        "brain coral",
        "flatworm",
        "nematode",
        "conch",
        "snail",
        "slug",
        "sea slug",
        "chiton",
        "chambered nautilus",
        "Dungeness crab",
        "rock crab",
        "fiddler crab",
        "king crab",
        "American lobster",
        "spiny lobster",
        "crayfish",
        "hermit crab",
        "isopod",
        "white stork",
        "black stork",
        "spoonbill",
        "flamingo",
        "little blue heron",
        "great egret",
        "bittern",
        "crane",
        "limpkin",
        "European gallinule",
        "American coot",
        "bustard",
        "ruddy turnstone",
        "red-backed sandpiper",
        "redshank",
        "dowitcher",
        "oystercatcher",
        "pelican",
        "king penguin",
        "albatross",
        "grey whale",
        "killer whale",
        "dugong",
        "sea lion",
        "Chihuahua",
        "Japanese spaniel",
        "Maltese dog",
        "Pekinese",
        "Shih-Tzu",
        "Blenheim spaniel",
        "papillon",
        "toy terrier",
        "Rhodesian ridgeback",
        "Afghan hound",
        "basset",
        "beagle",
        "bloodhound",
        "bluetick",
        "black-and-tan coonhound",
        "Walker hound",
        "English foxhound",
        "redbone",
        "borzoi",
        "Irish wolfhound",
        "Italian greyhound",
        "whippet",
        "Ibizan hound",
        "Norwegian elkhound",
        "otterhound",
        "Saluki",
        "Scottish deerhound",
        "Weimaraner",
        "Staffordshire bullterrier",
        "American Staffordshire terrier",
        "Bedlington terrier",
        "Border terrier",
        "Kerry blue terrier",
        "Irish terrier",
        "Norfolk terrier",
        "Norwich terrier",
        "Yorkshire terrier",
        "wire-haired fox terrier",
        "Lakeland terrier",
        "Sealyham terrier",
        "Airedale",
        "cairn",
        "Australian terrier",
        "Dandie Dinmont",
        "Boston bull",
        "miniature schnauzer",
        "giant schnauzer",
        "standard schnauzer",
        "Scotch terrier"
    ]
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    accuracy = zero_shot_classification(model, test_loader, class_names)
    for name, score in accuracy.items():
        print(f"{name}: {score / len(test_set.dataset)}")


if __name__ == "__main__":
    main()

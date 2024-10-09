from configs import CFG
import os
import requests
import zipfile
from pycocotools.coco import COCO
import torch
import cv2
import albumentations as A
import soundfile as sf


def download_dataset(data_dir="../datasets"):
    # Create caption and image directories
    annotations_dir = os.path.join(data_dir, "annotations")
    images_dir = os.path.join(data_dir, "train2014")

    # Download annotations (captions)
    zip_file = os.path.join(annotations_dir, "annotations.zip")
    url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    response = requests.get(url, stream=True)
    # write chunk in zip file
    with open(zip_file, "wb") as f:
        # 8192 = 8KB chunks (block or piece of data)
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    # unzip file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(data_dir)  # Extract all contents to the specified directory
    os.remove(zip_file)

    # Download images
    zip_file = os.path.join(images_dir, "train2014.zip")
    url = "http://images.cocodataset.org/zips/train2014.zip"
    response = requests.get(url, stream=True)
    # write chunk in zip file
    with open(zip_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    # unzip file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(data_dir)  # Extract all contents to the specified directory
    os.remove(zip_file)


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
    def __init__(self, images, audios, transformer, audio_processor, sample_rate=CFG.sample_rate,
                 duration_seconds=CFG.duration_seconds, *args, **kwargs):
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
        self.duration_seconds = duration_seconds

    def __getitem__(self, index):
        # Select caption in position index
        item = {}

        # image
        image = cv2.imread(f"{self.images[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transformer(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()

        # convert caption to audio
        audio_encoding = self.process_audio(self.audios[index])
        item["audio"] = audio_encoding["input_values"].squeeze(0)

        return item

    def process_audio(self, audio_name):
        # Load the audio file
        audio_input, sampling_rate = sf.read(audio_name)

        # Pad or truncate audio to a fixed length
        desired_length = self.duration_seconds * self.sample_rate
        current_length = len(audio_input)

        if current_length < desired_length:
            # Pad the audio data with zeros
            padding = torch.zeros((desired_length - current_length,))
            audio_input = torch.cat((torch.tensor(audio_input), padding))
        elif current_length > desired_length:
            # Truncate the audio data
            audio_input = audio_input[:desired_length]

        # Encode audio to fixed-size representation (you need to implement self.audio_processor)
        audio_encoding = self.audio_processor(audio_input, sampling_rate=self.sample_rate, return_tensors="pt")

        return audio_encoding

    def __len__(self):
        return len(self.audios)


# Resize and Normalize image
def get_transforms(mode="train"):
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


def build_loaders(dataframe, mode, audio_processor, sample_rate=CFG.sample_rate, duration_seconds=CFG.duration_seconds):
    transformer = get_transforms(mode=mode)
    dataset = DataGenerator(
        [img for img, _ in dataframe],
        [audio for _, audio in dataframe],
        transformer=transformer,
        audio_processor=audio_processor,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

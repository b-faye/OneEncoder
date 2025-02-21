from configs import CFG
import os
import requests
import zipfile
from pycocotools.coco import COCO
import torch
import cv2
import albumentations as A
import soundfile as sf



# Load Coco dataset
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

    # Download train images
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

    # Download val images
    images_dir = os.path.join(data_dir, "val2014")
    zip_file = os.path.join(images_dir, "val2014.zip")
    url = "http://images.cocodataset.org/zips/val2014.zip"
    response = requests.get(url, stream=True)
    # write chunk in zip file
    with open(zip_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    # unzip file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(data_dir)  # Extract all contents to the specified directory
    os.remove(zip_file)



# Build  image-caption pairs
def make_pairs(annotation_json_file, image_dir, max_captions=3):
    coco = COCO(annotation_json_file)
    image_ids = list(coco.imgs.keys())

    image_caption = []

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
                caption = annotation['caption']
                image_caption.append((image_path, caption))
                num_caption += 1

    return image_caption




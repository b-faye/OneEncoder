
################################################## PACKAGES ############################################################
################################################# PACKAGES #############################################################
# PyTorch data loading and utilities
import torch.multiprocessing

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


################################################### PARMETERS ##########################################################
################################################# PARAMETERS ###########################################################

class CFG:
    max_length = 128
    batch_size = 64
    num_workers = 4
    projection_dim = 256
    dropout_rate = 0.1
    num_head = 4
    num_layers = 1
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    modality_token_encoder_lr = 1e-3
    universal_projection_lr = 1e-3
    lr = 1e-3
    weight_decay = 1e-3
    patience = 5
    factor = 0.8
    token_size = 1
    epochs = 100
    image_size = 224
    token_dim = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_directory = "../multimodal/datasets"
    train_annotation_file = os.path.join(data_directory, "annotations", "captions_train2014.json")
    val_annotation_file = os.path.join(data_directory, "annotations", "captions_val2014.json")
    image_dir = os.path.join(data_directory, "train2014")
    image_dir_val = os.path.join(data_directory, "val2014")
    bert_name = "bert-base-uncased"
    vit_name = "vit_base_patch16_224"
print(f"Model is running on device: {CFG.device}")
######## DATASET ############################################################
################################################# DATASET  #############################################################

def download_coco_dataset(data_dir="../../datasets"):
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


def make_pairs(annotation_json_file, image_dir, max_captions=3, train=True):

    # Load coco captions
    coco = COCO(annotation_json_file)
    image_ids = list(coco.imgs.keys())

    coco_image_caption = []
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
                coco_image_caption.append((image_path, caption))
                num_caption += 1

    if train:
        # Flirk30K Dataset
        flirk_image_caption = []
        flirk_captions = pd.read_csv(os.path.join(CFG.data_directory, "flickr30k_images", "results.csv"), delimiter="|")
        flirk_captions.loc[flirk_captions[' comment'].isna(), " comment"] = (
            flirk_captions.loc)[flirk_captions[' comment'].isna(), " comment_number"]
        for index, row in flirk_captions.iterrows():
            image_path = os.path.join(CFG.data_directory, "flickr30k_images/flickr30k_images", row["image_name"])
            caption = row[" comment"]
            flirk_image_caption.append((image_path, caption))

        # TextCaps
        text_caps_image_caption = []
        text_caps_captions = pd.read_csv(os.path.join(CFG.data_directory, "TextCaps/textcaps_train.csv"))
        for index, row in text_caps_captions.iterrows():
            image_path = os.path.join(CFG.data_directory, "TextCaps/images_zip/images/train_val", row["image"])
            caption = row["answer"]
            text_caps_image_caption.append((image_path, caption))

        return coco_image_caption, flirk_image_caption, text_caps_image_caption

    else:
        # TextCaps
        text_caps_image_caption = []
        text_caps_captions = pd.read_csv(os.path.join(CFG.data_directory, "TextCaps/textcaps_val.csv"))
        for index, row in text_caps_captions.iterrows():
            image_path = os.path.join(CFG.data_directory, "TextCaps/images_zip/images/train_val", row["image"])
            caption = row["answer"]
            text_caps_image_caption.append((image_path, caption))

        return coco_image_caption, None, text_caps_image_caption

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, images, captions, tokenizer, transformer, *args, **kwargs):
        super(DataGenerator, self).__init__(*args, **kwargs)
        # Image processor
        self.transformer = transformer
        # List of images
        self.images = images
        # List of captions
        self.captions = captions
        # Tokenize all captions
        # Return: dict of keys "input_ids", "token_type_ids", and "attention_mask"
        # each value is list of lists (number of captions): [[], [], ..., []]
        self.tokenized_captions = tokenizer(
            list(self.captions), padding=True, truncation=True, max_length=CFG.max_length
        )

    def __getitem__(self, index):
        # Select caption in position index
        item = {
            k: torch.tensor(v[index]) for k, v in self.tokenized_captions.items()
        }
        # Select image in position index
        image = cv2.imread(f"{self.images[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transformer(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        # Caption text
        item["caption"] = self.captions[index]
        return item

    def __len__(self):
        return len(self.captions)


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

def build_loaders(dataframe, tokenizer, mode):
    transformer = get_transforms(mode=mode)
    dataset = DataGenerator(
        [img for img, _ in dataframe],
        [caption for _, caption in dataframe],
        tokenizer=tokenizer,
        transformer=transformer,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )

    return dataloader

################################################### MODELS ############################################################
################################################# MODELS ##############################################################

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=CFG.projection_dim, dropout_rate=CFG.dropout_rate, *args, **kwargs):
        """
        Projection Head module for contrastive learning.

        :param input_dim: Dimensionality of input features.
        :param projection_dim: Dimensionality of projected features (default: CFG.projection_dim).
        :param dropout_rate: Dropout rate (default: CFG.dropout_rate).
        """
        super(ProjectionHead, self).__init__(*args, **kwargs)

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
        """
        Forward pass of the projection head.

        :param inputs: Input features.
        :return: Projected features.
        """
        x = inputs
        x = self.linear_layer1(x)
        x = self.gelu(x)
        x = self.linear_layer2(x)
        x = self.dropout(x)
        x = self.normalization_layer(x)

        return x

    def __call__(self, inputs):
        """
        Callable method for the projection head.

        :param inputs: Input features.
        :return: Projected features.
        """
        return self.forward(inputs)


class ImageEncoder(nn.Module):
    def __init__(self, model_name=CFG.vit_name, projection_dim=CFG.projection_dim, trainable=False,
                 dropout_rate=CFG.dropout_rate, *args, **kwargs):
        """
        Image encoder module using Vision Transformer (ViT) backbone.

        :param model_name: Name of the Vision Transformer model (default: CFG.vit_name).
        :param projection_dim: Dimensionality of projected features (default: CFG.projection_dim).
        :param trainable: Whether to make the backbone trainable (default: False).
        :param dropout_rate: Dropout rate (default: CFG.dropout_rate).
        """
        super(ImageEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.trainable = trainable
        self.dropout_rate = dropout_rate
        # Models
        self.pretrained_vit = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self.projection_head = ProjectionHead(self.pretrained_vit.embed_dim, self.projection_dim, self.dropout_rate)
        # Freeze pretrained ViT layers
        for parameter in self.pretrained_vit.parameters():
            parameter.requires_grad = self.trainable

    def forward(self, images):
        """
        Forward pass of the image encoder.

        :param images: Input images.
        :return: Projected features.
        """
        x = images
        # forward_features: to return sequences (encoder) -> torch.Size([batch_size, 197, 768]) forward_head: to
        # return flattened sequences (vectors) -> torch.Size([batch_size, 768]) if num_classes=0 (no classification)
        # in timm.create_model and torch.Size([batch_size, 1000]) otherwise (classification)
        x = self.pretrained_vit.forward_features(x)
        # output: torch.Size([batch_size, 197, 256])
        x = self.projection_head(x)

        return x

    def __call__(self, images):
        """
        Callable method for the image encoder.

        :param images: Input images.
        :return: Projected features.
        """
        return self.forward(images)


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.bert_name, projection_dim=CFG.projection_dim,
                 trainable=False, dropout_rate=CFG.dropout_rate, *args, **kwargs):
        """
        Text encoder module using BERT backbone.

        :param model_name: Name of the BERT model (default: CFG.bert_name).
        :param projection_dim: Dimensionality of projected features (default: CFG.projection_dim).
        :param trainable: Whether to make the backbone trainable (default: False).
        :param dropout_rate: Dropout rate (default: CFG.dropout_rate).
        """
        super(TextEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        # Models
        self.pretrained_bert = BertModel.from_pretrained(self.model_name)
        self.projection_head = ProjectionHead(self.pretrained_bert.config.hidden_size,
                                              self.projection_dim, self.dropout_rate)
        # Freeze BERT
        for parameter in self.pretrained_bert.parameters():
            parameter.requires_grad = self.trainable

    def forward(self, captions):
        """
        Forward pass of the text encoder.

        :param captions: Input captions (input_ids, attention_mask).
        :return: Projected features.
        """
        input_ids, attention_mask = captions
        # last_hidden_state: torch.Size([batch_size, sequence, 768])
        # pooler_output: torch.Size([batch_size, 768])
        x = self.pretrained_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # output: torch.Size([batch_size, sequence, 256])
        x = self.projection_head(x)

        return x

    def __call__(self, captions):
        """
        Callable method for the text encoder.

        :param captions: Input captions (input_ids, attention_mask).
        :return: Projected features.
        """
        return self.forward(captions)


class ModalityTokenEncoder(nn.Module):
    def __init__(self, projection_dim=CFG.projection_dim, token_size=CFG.token_size, device='cpu', token_dim=CFG.token_dim, *args, **kwargs):
        """
        Modality token encoder module for encoding modality token information.

        :param projection_dim: Dimensionality of projected features (default: CFG.projection_dim).
        :param token_size: Token size.
        :param device: Device to run the module on (default: 'cpu').
        :param token_dim: Dimension of tokens
        """
        super(ModalityTokenEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.projection_dim = projection_dim
        self.device = device
        self.token_size = token_size
        self.token_dim = token_dim
        # Models
        text_variance = torch.rand(1) * 0.5 + 0.1
        image_variance = torch.rand(1) * 0.5 + 0.1
        self.text_token = nn.Parameter(torch.normal(mean=0, std=text_variance.item(),
                                                      size=(self.token_size, self.token_dim)).to(self.device))
        self.image_token = nn.Parameter(torch.normal(mean=0, std=image_variance.item(),
                                                       size=(self.token_size, self.token_dim)).to(self.device))
        # Projection
        self.token_projection = nn.Sequential(
            nn.Linear(self.token_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        )

    def forward(self, modality_type):
        """
        Forward pass of the modality encoder.

        :param modality_type: Input token indicator.
        :return: Projected features.
        """
        token = torch.where(torch.tensor(modality_type == "image"), self.image_token, self.text_token)
        token_features = self.token_projection(token)
        return token_features

    def __call__(self, modality_type):
        """
        Callable method for the token encoder.

        :param modality_type: Input token indicator.
        :return: Projected features.
        """
        return self.forward(modality_type)


class UniversalProjectionOutput:
    def __init__(self, outputs):
        """
        Wrapper class for projection model outputs.

        :param outputs: Dictionary containing model outputs.
        """
        self.outputs = outputs

    def __getattr__(self, name):
        """
        Retrieve attribute from outputs dictionary.

        :param name: Name of the attribute to retrieve.
        :return: Value of the attribute.
        """
        if name in self.outputs:
            return self.outputs[name]
        else:
            raise AttributeError(f"'UniversalProjectionOutput' object has no attribute '{name}'")


class UniversalProjectionEncoder(nn.Module):
    def __init__(self, input_dim=CFG.projection_dim, num_head=CFG.num_head, num_layers=CFG.num_layers, *args, **kwargs):
        """
         Initialize Universal Projection module.

            :param input_dim: Dimensionality of the input embeddings. Defaults to CFG.projection_dim.
            :param num_head: Number of attention heads. Defaults to CFG.num_head.
            :param num_layers: Number of transformer layers. Defaults to CFG.num_layers.
        """
        super(UniversalProjectionEncoder, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.num_head = num_head
        self.num_layers = num_layers

        self.transformer_encoder_block = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=self.num_head,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_block,
            num_layers=self.num_layers
        )

        # self.transformer_encoder = TransformerModel(self.input_dim, self.num_head, self.num_layers)

        # model_name = 'bert-large-uncased'
        self.layer_normalization = nn.LayerNorm(self.input_dim)
        # self.transfopip install torch torchvision -Urmer_encoder = BertModel.from_pretrained(model_name)

    def forward(self, inputs):
        # x: image or caption embeddings
        x, tokens = inputs

        ## Universal Projection block
        tokens = tokens.unsqueeze(0).expand(x.size()[0], -1, -1)

        # Concatenate tokens with image/caption embeddings
        output_tensor = x * tokens

        # Normalization
        output_norm = self.layer_normalization(output_tensor)

        # Projection
        output_encoder = self.transformer_encoder(output_norm)

        ## Residual Connection
        residual_output = output_encoder + output_tensor

        # Residual connection
        return UniversalProjectionOutput({'last_hidden_state': residual_output,
                             'mean_output': torch.mean(residual_output, dim=1),
                             'pooler_output': residual_output[:, 0, :]})

    def __call__(self, inputs):
        return self.forward(inputs)

class OneEncoder(nn.Module):

    def __init__(self, image_encoder=ImageEncoder(), text_encoder=TextEncoder(),
                 modality_token_encoder=ModalityTokenEncoder(),
                 universal_projection_encoder=UniversalProjectionEncoder(), device='cpu',
                 tokenizer=BertTokenizer.from_pretrained(CFG.bert_name),
                 image_preprocessor=A.Compose([A.Resize(CFG.image_size, CFG.image_size, always_apply=True),
                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True), ToTensorV2()]),
                 *args, **kwargs):
        """
        Initialize the model.

        :param image_encoder: Image encoder module (default: ImageEncoder()).
        :param text_encoder: Text encoder module (default: TextEncoder()).
        :param modality_token_encoder: Modality encoder module (default: ModalityEncoder()).
        :param universal_projection_encoder: Universal projection encoder module (default: UniversalProjection()).
        :param device: Device to run the model on (default: 'cpu').
        :param tokenizer: Tokenizer for text encoding (default: BertTokenizer.from_pretrained(CFG.bert_name)).
        :param image_preprocessor: Preprocessor for image inputs (default: A.Compose([...])).
        """

        super(OneEncoder, self).__init__(*args, **kwargs)
        self.device = device
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.universal_projection_encoder = universal_projection_encoder
        self.modality_token_encoder = modality_token_encoder
        self.modality_token_encoder.device = self.device
        self.tokenizer = tokenizer
        self.image_preprocessor = image_preprocessor
        # The learnable temperature parameter τ was initialized to the equivalent of 0.07 from (Wu et al., 2018)
        # and clipped to prevent scaling the logits by more than 100, which we found necessary
        # to prevent training instability.
        self.temperature = nn.Parameter(torch.tensor(0.07).to(self.device))

    @classmethod
    def load_image(cls, image_path):
        # Load online image
        if image_path.startswith("http"):
            response = requests.get(image_path)
            # Check if the request was successful
            if response.status_code == 200:
                # Convert the image content to a numpy array
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)

                # Decode the image using OpenCV
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Load local image
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def encode_image(self, image_paths=None, image_tensors=None, outputs="mean"):
        """
        Encode images into feature vectors.

        :param image_paths: List of image paths.
        :param image_tensors: Torch tensor (batch, 3, 224, 224).
        :param outputs type of outputs: mean, pooler, sequence
        :return: Encoded image features.
        """
        if image_paths is not None:
            image_processed = [self.image_preprocessor(image=self.load_image(image))["image"] for image in image_paths]
            image_processed = torch.stack(image_processed).to(self.device)
            with torch.no_grad():
                image_features = self.image_encoder(image_processed.to(self.device))
                modality_token_feature = self.modality_token_encoder("image")
                output_features = self.universal_projection_encoder([image_features, modality_token_feature])

        elif image_tensors is not None:
            with torch.no_grad():
                image_features = self.image_encoder(image_tensors.to(self.device))
                modality_token_feature = self.modality_token_encoder("image")
                output_features = self.universal_projection_encoder([image_features, modality_token_feature])
        if outputs == "mean":
            image_features = output_features.mean_output
        elif outputs == "sequence":
            image_features = output_features.last_hidden_state
        else:
            image_features = output_features.pooler_output

        return image_features

    def encode_text(self, texts, max_length=128, outputs="mean"):
        """
        Encode text descriptions into feature vectors.

        :param texts: List of text descriptions.
        :param max_length: Maximum length of the text sequences (default: 128).
        :param outputs type of outputs: mean, sequence, pooler
        :return: Encoded text features.
        """
        encoded_query = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length
        )
        batch = {
            key: torch.tensor(values).to(self.device)
            for key, values in encoded_query.items()
        }
        with torch.no_grad():
            text_features = self.text_encoder([
                batch["input_ids"], batch["attention_mask"]
            ])
            modality_token_feature = self.modality_token_encoder("text")
            output_features = self.universal_projection_encoder([text_features, modality_token_feature])
            if outputs == "mean":
                text_features = output_features.mean_output
            elif outputs == "sequence":
                text_features = output_features.last_hidden_state
            else:
                text_features = output_features.last_hidden_state

        return text_features

    def matching(self, image_paths, texts, normalize=True, top_k=None, strategy="similarity", temperature=1.0):
        """
        Calculate similarities between images and texts.

        :param image_paths: List of paths to images.
        :param texts: List of text descriptions.
        :param normalize: Whether to normalize the features (default: True).
        :param top_k: Return top K results (default: None).
        :param strategy: Matching strategy, either 'similarity' or 'softmax' (default: 'similarity').
        :param temperature: change real distribution, default = 2.5
        :return: If top_k is provided, returns top probabilities and labels, otherwise returns dot similarities.
        """

        image_features = self.encode_image(image_paths=image_paths)
        text_features = self.encode_text(texts=texts)

        if normalize:
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
        dot_similarities = (image_features @ text_features.T) * torch.exp(torch.tensor(temperature).to(self.device))
        if strategy == 'softmax':
            dot_similarities = (float(len(set(texts))) * dot_similarities).softmax(dim=-1)
        if top_k is not None:
            top_probs, top_labels = dot_similarities.cpu().topk(top_k, dim=-1)
            return top_probs, top_labels
        else:
            return dot_similarities, None

    def image_retrieval(self, query, image_paths, image_embeddings=None, temperature=1.0, n=9, plot=False):
        """
        Perform image retrieval based on a text query.

        :param query: Text query (string).
        :param image_paths: List of image paths (optional).
        :param image_embeddings: Precomputed image embeddings (optional).
        :param temperature: change real distribution, default = 2.5
        :param n: Number of images to retrieve (default: 9).
        :param plot: Whether to plot the retrieved images (default: False).
        :return: Tuple containing similarity values and indices of the retrieved images.
        """
        text_embeddings = self.encode_text([query])
        if image_embeddings is None:
            image_embeddings = self.encode_image(image_paths=image_paths)

        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = (text_embeddings_n @ image_embeddings_n.T) * torch.exp(
            torch.tensor(temperature).to(self.device))
        if n > len(image_paths):
            n = len(image_paths)
        values, indices = torch.topk(dot_similarity.cpu().squeeze(0), n)
        if plot:
            nrows = int(np.sqrt(n))
            ncols = int(np.ceil(n / nrows))
            matches = [image_paths[idx] for idx in indices]
            fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
            for match, ax in zip(matches, axes.flatten()):
                image = self.load_image(f"{match}")
                ax.imshow(image)
                ax.axis("off")

            fig.suptitle(query)
            plt.show()
        return values, indices

    def text_retrieval(self, query, texts, text_embeddings=None, n=9, plot_image=False, temperature=1.0):
        """
        Perform text retrieval based on an image query.

        :param query: Image query (path of image).
        :param texts: List of text samples.
        :param text_embeddings: Precomputed text embeddings (optional).
        :param n: Number of texts to retrieve (default: 9).
        :param plot_image: Plot the query
        :param temperature: change real distribution, default = 2.5
        :return: List of retrieved text samples and its probabilities.
        """
        if text_embeddings is None:
            text_embeddings = self.encode_text(texts)

        image_embeddings = self.encode_image([query])
        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = (image_embeddings_n @ text_embeddings_n.T) * torch.exp(
            torch.tensor(temperature).to(self.device))

        if n > len(texts):
            n = len(texts)

        values, indices = torch.topk(dot_similarity.cpu().squeeze(0), n)
        matches = [texts[idx] for idx in indices]
        if plot_image:
            # Read and plot the image
            image = self.load_image(query)
            # Plot the image
            plt.imshow(image)
            plt.title('Random Image')
            plt.axis('off')
            plt.show()
        return matches, values

    def forward(self, inputs):
        """
        Forward pass of the model.

        :param inputs: Input dictionary containing 'image', 'input_ids', and 'attention_mask'.
        :return: Loss value.
        """

        images = inputs['image']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # For constant temperature replace self.temperature by self.temperature.data and give the target temperature
        # for example temperature = 1.0 or temperature = 2.5
        t = torch.clamp(self.temperature.data, min=torch.tensor(2.5).to(self.device),
                        max=torch.tensor(3.0).to(self.device))

        # Embeddings
        image_features = self.image_encoder(images)
        text_features = self.text_encoder([input_ids, attention_mask])
        image_modality_token = self.modality_token_encoder("image")
        text_modality_token = self.modality_token_encoder("text")

        # Universal Projection
        image_features = self.universal_projection_encoder([image_features, image_modality_token]).mean_output
        text_features = self.universal_projection_encoder([text_features, text_modality_token]).mean_output

        # L2 normalization
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Compute loss
        logits = (image_features @ text_features.T) * torch.exp(t)
        labels = torch.arange(logits.size(0)).to(self.device)
        loss_i = F.cross_entropy(logits, labels, reduction='mean')
        loss_t = F.cross_entropy(logits.t(), labels, reduction='mean')
        loss = (loss_i + loss_t) / 2.0

        return loss

    def __call__(self, inputs):
        return self.forward(inputs)

################################################### DEFINE TRAINING METHODS ############################################
################################################# DEFINE TRAINING METHODS ##############################################
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

        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

################################################## SAVE EMBEDDINGS #####################################################
################################################# SAVE EMBEDDINGS ######################################################

def get_image_embeddings(pairs, model_path):
    # Process pairs to delete duplicated images
    unique_images = set()
    unique_pairs = [(item[0], item[1]) for item in pairs if item[0] not in unique_images
                    and not unique_images.add(item[0])]
    # sort images
    unique_pairs = sorted(unique_pairs, key=lambda x: x[0])
    # Build model
    model = OneEncoder(device=CFG.device)
    # Load parameters
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    valid_loader = build_loaders(unique_pairs, model.tokenizer, mode="valid")
    # Use eval mode to freeze all layers
    model.eval()
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            token_embedding = model.modality_token_encoder("image")
            image_embeddings = model.image_encoder(batch["image"].to(CFG.device))
            output_embeddings = model.universal_projection_encoder([image_embeddings, token_embedding]).mean_output
            valid_image_embeddings.append(output_embeddings)

    return torch.cat(valid_image_embeddings)


def get_caption_embeddings(pairs, model_path):
    # sort according images
    unique_pairs = sorted(pairs, key=lambda x: x[0])
    # Build model
    model = OneEncoder(device=CFG.device)
    # Load parameters
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    valid_loader = build_loaders(unique_pairs, model.tokenizer, mode="valid")
    # Use eval mode to freeze all layers
    model.eval()
    valid_caption_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            token_embedding = model.modality_token_encoder("text")
            caption_embeddings = model.text_encoder([batch["input_ids"].to(CFG.device),
                                                   batch["attention_mask"].to(CFG.device)])
            output_embeddings = model.universal_projection_encoder([caption_embeddings, token_embedding]).mean_output
            valid_caption_embeddings.append(output_embeddings)
    return torch.cat(valid_caption_embeddings)


################################################### TRAINING STEP ######################################################
################################################# TRAINING STEP ########################################################

def main(basic_train=False):
    # Load the pretrained Bert tokenizer
    tokenizer = BertTokenizer.from_pretrained(CFG.bert_name)

    # Load COCO dataset if not exist
    if len(os.listdir(CFG.data_directory)) == 0:
        download_coco_dataset()

    # Create pairs image-caption 413.915
    training_pairs, _, _ = make_pairs(CFG.train_annotation_file, CFG.image_dir, 5)
    random.shuffle(training_pairs)
    # validation 202.520
    validation_pairs, _, _ = make_pairs(CFG.val_annotation_file, CFG.image_dir_val, 5)
    random.shuffle(validation_pairs)
    validation_pairs = validation_pairs[-round(len(validation_pairs)*0.20):]
    print("Number of training images: {}".format(len(training_pairs)))
    print("Number of validation images: {}".format(len(validation_pairs)))

    # Build loader : return dictionary
    train_loader = build_loaders(training_pairs, tokenizer, mode="train")
    val_loader = build_loaders(validation_pairs, tokenizer, mode="valid")

    # Build the model
    model = OneEncoder(device=CFG.device)
    model = model.to(CFG.device)

    if basic_train:
        # Train all parameters with the same lr and weight decay
        # This method is better when using dynamique temperature
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=CFG.weight_decay, lr=CFG.lr)
    else:
        parameters = [
            {"params": model.image_encoder.pretrained_vit.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.text_encoder.pretrained_bert.parameters(), "lr": CFG.text_encoder_lr},
            {
                "params": itertools.chain(
                    model.modality_token_encoder.parameters(), model.universal_projection_encoder.parameters(),
                    model.image_encoder.projection_head.parameters(), model.text_encoder.projection_head.parameters()
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

    # Train the model
    for epoch in range(CFG.epochs):
        print(model.temperature.data)
        print("Epoch: %d" % (epoch+1))
        # Set the model in train mode
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, "epoch")
        print(f"Epoch: {epoch+1}, train loss: {train_loss}")
        # Set the model in evaluation mode
        model.eval()
        with torch.no_grad():
            val_loss = valid_epoch(model, val_loader)
            print(f"Epoch: {epoch + 1}, val loss: {val_loss}")
        if val_loss.avg < best_loss:
            best_loss = val_loss.avg
            num_bad_epochs = 0
            torch.save(model.state_dict(), "best.pt")
            best_epoch = epoch+1
            print("Saved best model!")
        else:
            if epoch >= CFG.patience - 1:
                num_bad_epochs += 1
            if num_bad_epochs >= CFG.patience:
                print(f"Early stopping at epoch {epoch + 1}. Restoring best weights...")
                break
        lr_scheduler.step(val_loss.avg)
    torch.save(model.state_dict(), "last.pt")

    # Save train embeddings with best.pt
    image_embeddings = get_image_embeddings(training_pairs, "best.pt")
    torch.save(image_embeddings, "image_embeddings_best.pt")
    caption_embeddings = get_caption_embeddings(training_pairs, "best.pt")
    torch.save(caption_embeddings, "caption_embeddings_best.pt")
    # Free GPU
    model = None
    optimizer = None
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

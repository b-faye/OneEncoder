
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
from configs import CFG

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

class ScaledDotProduct(nn.Module):
    def __init__(self, projection_dim=CFG.projection_dim, *args, **kwargs):
        super(ScaledDotProduct, self).__init__(*args, **kwargs)
        self.projection_dim = projection_dim
        self.query = nn.Linear(self.projection_dim, self.projection_dim)
        self.key = nn.Linear(self.projection_dim, self.projection_dim)
        self.value = nn.Linear(self.projection_dim, self.projection_dim)

    def forward(self, inputs):
        keys = self.key(inputs[0])
        values = self.value(inputs[0])
        query = self.query(inputs[1])

        # Compute scaled dot product
        scores = torch.bmm(query, keys.transpose(1, 2))  # Batch matrix multiplication
        scaling_factor = self.projection_dim ** 0.5  # Scaling factor
        scores /= scaling_factor  # Apply scaling factor
        attention_weights = F.softmax(scores, dim=-1)

        # Compute context vector
        context_vector = torch.bmm(attention_weights, values)

        return context_vector

    def __call__(self, inputs):
        return self.forward(inputs)

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

        self.scaled_dot_product = ScaledDotProduct(projection_dim=self.input_dim)
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

        attention = self.scaled_dot_product([x, tokens])
        
        output_tensor = x + attention

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

    def matching(self, image_paths, texts, normalize=True, top_k=None, strategy="similarity", temperature=2.5):
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

    def image_retrieval(self, query, image_paths, image_embeddings=None, temperature=2.5, n=9, plot=False):
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

    def text_retrieval(self, query, texts, text_embeddings=None, n=9, plot_image=False, temperature=2.5):
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


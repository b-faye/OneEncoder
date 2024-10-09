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


# COCO dataset tools
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

from configs import CFG
from costum_datasets import download_dataset, make_pairs, build_loaders

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=CFG.projection_dim, dropout_rate=CFG.dropout_rate, *args, **kwargs):
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
        x = inputs
        x = self.linear_layer1(x)
        x = self.gelu(x)
        x = self.linear_layer2(x)
        x = self.dropout(x)
        x = self.normalization_layer(x)
        return x

    def __call__(self, inputs):
        return self.forward(inputs)


class ImageEncoder(nn.Module):
    def __init__(self, model_name=CFG.vit_name, projection_dim=CFG.projection_dim, trainable=False,
                 dropout_rate=CFG.dropout_rate, *args, **kwargs):
        super(ImageEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.trainable = trainable
        self.dropout_rate = dropout_rate
        # Models
        self.pretrained_vit = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self.projection_head = ProjectionHead(self.pretrained_vit.embed_dim, self.projection_dim, self.dropout_rate)
        # Freeze pretrained vit layers
        for parameter in self.pretrained_vit.parameters():
            parameter.requires_grad = self.trainable

    def forward(self, images):
        x = images
        # forward_features: to return sequences (encoder) -> torch.Size([batch_size, 197, 768]) forward_head: to
        # return flattened sequences (vectors) -> torch.Size([batch_size, 768]) if num_classes=0 (no classification)
        # in timm.create_model and torch.Size([batch_size, 1000]) otherwise (classification)
        x = self.pretrained_vit.forward_features(x)
        # output: torch.Size([batch_size, 197, 256])
        x = self.projection_head(x)
        return x

    def __call__(self, images):
        return self.forward(images)


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.bert_name, projection_dim=CFG.projection_dim,
                 trainable=False, dropout_rate=CFG.dropout_rate, *args, **kwargs):
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
        # Freeze bert
        for parameter in self.pretrained_bert.parameters():
            parameter.requires_grad = self.trainable

    def forward(self, captions):
        input_ids, attention_mask = captions
        # last_hidden_state: torch.Size([batch_size, sequence, 768])
        # pooler_output: torch.Size([batch_size, 768])
        x = self.pretrained_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # output: torch.Size([batch_size, sequence, 256])
        x = self.projection_head(x)
        return x

    def __call__(self, captions):
        return self.forward(captions)


class ContextEncoder(nn.Module):
    def __init__(self, input_dim=CFG.context_input_dim, projection_dim=CFG.projection_dim,
                 dropout_rate=CFG.dropout_rate, device='cpu', *args, **kwargs):
        super(ContextEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.context_input_dim = input_dim
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.device = device
        # Models
        text_variance = torch.rand(1) * 0.5 + 0.1
        image_variance = torch.rand(1) * 0.5 + 0.1
        self.text_context = nn.Parameter(torch.normal(mean=0, std=text_variance.item(),
                                                      size=(1, self.context_input_dim)).to(self.device))
        self.image_context = nn.Parameter(torch.normal(mean=0, std=image_variance.item(),
                                                       size=(1, self.context_input_dim)).to(self.device))
        self.model = nn.Sequential(
            nn.Linear(self.context_input_dim, 64),  # Adjust based on your requirements
            nn.ReLU(),
            nn.Linear(64, 128),  # Adjust based on your requirements
            nn.ReLU(),
            nn.Linear(128, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        )

    def forward(self, inputs):
        # 0: text and 1: image
        x = inputs
        context = torch.where(x == 0, self.text_context, self.image_context)
        # output : torch.Size([batch_size, 256])
        return self.model(context)

    def __call__(self, inputs):
        return self.forward(inputs)


class FusionOutput:
    def __init__(self, outputs):
        self.outputs = outputs

    def __getattr__(self, name):
        if name in self.outputs:
            return self.outputs[name]
        else:
            raise AttributeError(f"'FusionOutput' object has no attribute '{name}'")


class Fusion(nn.Module):
    def __init__(self, input_dim=CFG.projection_dim, num_head=CFG.num_head, num_layers=CFG.num_layers, *args, **kwargs):
        super(Fusion, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.num_head = num_head
        self.num_layers = num_layers
        self.projection_dim = input_dim
        # Models and layers
        self.transformer_encoder_block = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=self.num_head,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_block,
            num_layers=self.num_layers
        )

        self.layer_normalization = nn.LayerNorm(self.projection_dim)

    def forward(self, inputs):
        # x: image or caption embeddings
        x, contexts = inputs

        ## Fusion block
        addition = x + contexts.unsqueeze(1)

        # Normalization
        norm = self.layer_normalization(addition)

        # Projection
        encoder_output = self.transformer_encoder(norm)

        # Residual connection
        residual_connection = addition + encoder_output

        # Possible outputs
        sequence_outputs = residual_connection
        average_outputs = torch.mean(residual_connection, dim=1)
        max_outputs = torch.max(residual_connection, dim=1)
        min_outputs = torch.min(residual_connection, dim=1)
        return FusionOutput({'sequence_outputs': sequence_outputs, 'average_outputs': average_outputs,
                             'max_outputs': max_outputs, 'min_outputs': min_outputs})

    def __call__(self, inputs):
        return self.forward(inputs)


class OneEncoderUP(nn.Module):

    def __init__(self, image_encoder=ImageEncoder(), text_encoder=TextEncoder(), context_encoder=ContextEncoder(),
                 fusion_encoder=Fusion(), device='cpu', tokenizer=BertTokenizer.from_pretrained(CFG.bert_name),
                 image_preprocessor=A.Compose([A.Resize(CFG.image_size, CFG.image_size, always_apply=True),
                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True), ToTensorV2()]),
                 *args, **kwargs):
        """
        Initialize the model.

        :param image_encoder: Image encoder module (default: ImageEncoder()).
        :param text_encoder: Text encoder module (default: TextEncoder()).
        :param context_encoder: Context encoder module (default: ContextEncoder()).
        :param fusion_encoder: Fusion encoder module (default: Fusion()).
        :param device: Device to run the model on (default: 'cpu').
        :param tokenizer: Tokenizer for text encoding (default: BertTokenizer.from_pretrained(CFG.bert_name)).
        :param image_preprocessor: Preprocessor for image inputs (default: A.Compose([...])).
        """

        super(OneEncoderUP, self).__init__(*args, **kwargs)
        self.device = device
        self.to(self.device)
        self.image_encoder = image_encoder.to(self.device)
        self.text_encoder = text_encoder.to(self.device)
        self.context_encoder = context_encoder.to(self.device)
        self.context_encoder.device = self.device
        self.fusion_encoder = fusion_encoder.to(self.device)
        self.tokenizer = tokenizer
        self.image_preprocessor = image_preprocessor
        self.image_context_id = 1
        self.text_context_id = 0
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

    def encode_image(self, image_paths=None, image_tensors=None, outputs="average"):
        """
        Encode images into feature vectors.

        :param image_paths: List of image paths.
        :param image_tensors: Torch tensor (batch, 3, 224, 224).
        :param outputs type of outputs: average, min, max or sequence
        :return: Encoded image features.
        """
        if image_paths is not None:
            image_processed = [self.image_preprocessor(image=self.load_image(image))["image"] for image in image_paths]
            image_processed = torch.stack(image_processed).to(self.device)
            with torch.no_grad():
                image_features = self.image_encoder(image_processed.to(self.device))
                image_context_feature = self.context_encoder(torch.tensor([self.image_context_id],
                                                                    dtype=torch.float32).unsqueeze(1).to(self.device))
                image_context_features = image_context_feature.repeat(image_processed.size(0), 1)
                output_features = self.fusion_encoder([image_features, image_context_features])

        elif image_tensors is not None:

            with torch.no_grad():
                image_features = self.image_encoder(image_tensors.to(self.device))
                image_context_feature = self.context_encoder(
                    torch.tensor([self.image_context_id], dtype=torch.float32).unsqueeze(1).to(self.device))
                image_context_features = image_context_feature.repeat(image_tensors.size(0), 1)
                output_features = self.fusion_encoder([image_features, image_context_features])
        if outputs == "average":
            image_features = output_features.average_outputs
        elif outputs == "min":
            image_features = output_features.min_outputs
        elif outputs == "max":
            image_features = output_features.max_outputs
        else:
            image_features = output_features.sequence_outputs

        return image_features

    def encode_text(self, texts, max_length=128, outputs="average"):
        """
        Encode text descriptions into feature vectors.

        :param texts: List of text descriptions.
        :param max_length: Maximum length of the text sequences (default: 128).
        :param outputs type of outputs: average, min, max or sequence
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
            text_context_feature = self.context_encoder(
                torch.tensor([self.text_context_id], dtype=torch.float32).unsqueeze(1).to(self.device))
            text_context_features = text_context_feature.repeat(len(texts), 1)
            output_features = self.fusion_encoder([text_features, text_context_features])
            if outputs == "average":
                text_features = output_features.average_outputs
            elif outputs == "min":
                text_features = output_features.min_outputs
            elif outputs == "max":
                text_features = output_features.max_outputs
            else:
                text_features = output_features.sequence_outputs
        return text_features

    def matching(self, image_paths, texts, normalize=True, top_k=None, strategy="similarity"):
        """
        Calculate similarities between images and texts.

        :param image_paths: List of paths to images.
        :param texts: List of text descriptions.
        :param normalize: Whether to normalize the features (default: True).
        :param top_k: Return top K results (default: None).
        :param strategy: Matching strategy, either 'similarity' or 'softmax' (default: 'similarity').
        :return: If top_k is provided, returns top probabilities and labels, otherwise returns dot similarities.
        """
        image_features = self.encode_image(image_paths=image_paths)
        text_features = self.encode_text(texts=texts)

        if normalize:
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)

        dot_similarities = image_features @ text_features.T
        if strategy == 'softmax':
            dot_similarities = (float(len(set(texts))) * dot_similarities).softmax(dim=-1)
        if top_k is not None:
            top_probs, top_labels = dot_similarities.cpu().topk(top_k, dim=-1)
            return top_probs, top_labels
        else:
            return dot_similarities, None

    def image_retrieval(self, query, image_paths, image_embeddings=None, max_length=128, n=9, plot=False):
        """
        Perform image retrieval based on a text query.

        :param query: Text query (string).
        :param image_paths: List of image paths (optional).
        :param image_embeddings: Precomputed image embeddings (optional).
        :param max_length: Maximum length of the text sequence (default: 128).
        :param n: Number of images to retrieve (default: 9).
        :param plot: Whether to plot the retrieved images (default: False).
        :return: Tuple containing similarity values and indices of the retrieved images.
        """
        text_embeddings = self.encode_text([query])
        if image_embeddings is None:
            image_embeddings = self.encode_image(image_paths=image_paths)

        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = text_embeddings_n @ image_embeddings_n.T
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

    def text_retrieval(self, query, texts, text_embeddings=None, n=9, plot_image=False):
        """
        Perform text retrieval based on an image query.

        :param query: Image query (path of image).
        :param texts: List of text samples.
        :param text_embeddings: Precomputed text embeddings (optional).
        :param n: Number of texts to retrieve (default: 9).
        :param plot_image: Plot the query
        :return: List of retrieved text samples and its probabilities.
        """
        if text_embeddings is None:
            text_embeddings = self.encode_text(texts)

        image_embeddings = self.encode_image([query])
        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = image_embeddings_n @ text_embeddings_n.T

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
        t = torch.clamp(self.temperature, min=torch.tensor(0.01).to(self.device),
                        max=torch.tensor(20).to(self.device))

        # Embeddings
        image_features = self.image_encoder(images)
        text_features = self.text_encoder([input_ids, attention_mask])
        image_context_feature = self.context_encoder(torch.tensor([self.image_context_id],
                                                                  dtype=torch.float32).unsqueeze(1).to(self.device))
        text_context_feature = self.context_encoder(torch.tensor([self.text_context_id],
                                                                 dtype=torch.float32).unsqueeze(1).to(self.device))
        # repeat: images.size for the first dim and 1 for the second dim (like tile in tf)
        image_context_features = image_context_feature.repeat(images.size(0), 1)
        text_context_features = text_context_feature.repeat(images.size(0), 1)
        # Fusion
        image_features = self.fusion_encoder([image_features, image_context_features]).average_outputs
        text_features = self.fusion_encoder([text_features, text_context_features]).average_outputs
        # L2 normalization
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        # Compute loss
        logits = (image_features @ text_features.T) * torch.exp(t)
        # labels
        labels = torch.arange(logits.size(0)).to(self.device)
        loss_i = F.cross_entropy(logits, labels, reduction='mean')
        loss_t = F.cross_entropy(logits.t(), labels, reduction='mean')
        loss = (loss_i + loss_t) / 2.0

        return loss

    def __call__(self, inputs):
        return self.forward(inputs)


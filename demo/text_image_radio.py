# PyTorch for deep learning operations
import torch
import torch.nn as nn

# PyTorch data loading and utilities
import torch.multiprocessing

# COCO dataset tools
from transformers import BertModel, BertTokenizer, AutoModel, AutoImageProcessor

from configs import CFG
from text_image import OneEncoder as TextImageEncoder



class AlignmentLayer(nn.Module):
    def __init__(self, input_dim=768, projection_dim=CFG.projection_dim, dropout_rate=CFG.dropout_rate, *args,
                 **kwargs):
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


class RadioEncoder(nn.Module):
    def __init__(self, model_name=CFG.radio_name, projection_dim=CFG.projection_dim,
                 trainable=False, dropout_rate=CFG.dropout_rate, *args, **kwargs):
        super(RadioEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        # Models
        self.pretrained_encoder = AutoModel.from_pretrained(self.model_name)
        self.alignment_layer = AlignmentLayer(
            input_dim=self.pretrained_encoder.config.hidden_size,
            projection_dim=self.projection_dim,
            dropout_rate=self.dropout_rate)
        # Freeze Wav2VecModel
        for parameter in self.pretrained_encoder.parameters():
            parameter.requires_grad = self.trainable

    def forward(self, inputs):
        x = self.pretrained_encoder(inputs).last_hidden_state
        x = self.alignment_layer(x)
        return x

    def __call__(self, inputs):
        return self.forward(inputs)


class ModalityTokenEncoder(nn.Module):
    def __init__(self, projection_dim=CFG.projection_dim, token_size=CFG.token_size, device='cpu', token_dim=CFG.token_dim, *args, **kwargs):
        super(ModalityTokenEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.projection_dim = projection_dim
        self.device = device
        self.token_size = token_size
        self.token_dim = token_dim
        # Models
        audio_variance = torch.rand(1) * 0.5 + 0.1
        self.radio_token = nn.Parameter(torch.normal(mean=0, std=audio_variance.item(),
                                                      size=(self.token_size, self.token_dim)).to(self.device))

        self.token_projection = nn.Sequential(
            nn.Linear(self.token_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        )

    def forward(self):
        return self.token_projection(self.radio_token)

    def __call__(self):
        return self.forward()


class OneEncoder(nn.Module):
    def __init__(self, device='cpu', modality_token_encoder=ModalityTokenEncoder(),
                 checkpoint="bilalfaye/OneEncoder-text-image",
                 radio_processor=AutoImageProcessor.from_pretrained("microsoft/rad-dino"),
                 sample_rate=CFG.sample_rate, radio_encoder=RadioEncoder(), *args, **kwargs):
        super(OneEncoder, self).__init__(*args, **kwargs)

        self.device = device
        self.checkpoint = checkpoint
        self.modality_token_encoder = modality_token_encoder
        self.modality_token_encoder.device = self.device
        self.text_image_encoder = TextImageEncoder(device=self.device)
        self.text_image_encoder.from_pretrained(self.checkpoint)
        self.radio_processor = radio_processor
        self.sample_rate = sample_rate
        self.radio_encoder = radio_encoder
        self.temperature = nn.Parameter(torch.tensor(0.07).to(self.device))

        # Freeze
        for parameter in self.text_image_encoder.parameters():
            parameter.requires_grad = False

    def encode_radio(self, pil_radios=None, radios=None):
        """
        :param pil_radios: list of pillow images
        :param radios: preprocessed image
        :return: tensor
        """
        if pil_radios is not None:
            tensors = self.radio_processor(pil_radios, return_tensors="pt")["pixel_values"].to(self.device)
        else:
            tensors = radios.to(self.device)
        features = self.radio_encoder(tensors)
        radio_token = self.modality_token_encoder()
        outputs = self.text_image_encoder.universal_projection_encoder([features, radio_token]).last_hidden_state
        return outputs


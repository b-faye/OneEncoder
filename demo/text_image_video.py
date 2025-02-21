################################################## PACKAGES ############################################################
################################################# PACKAGES #############################################################
# PyTorch for deep learning operations
import torch
import torch.nn as nn


# PyTorch data loading and utilities
import torch.multiprocessing

# Additional PyTorch modules and libraries
import numpy as np

# Hugging Face Transformers library for BERT models
from transformers import BertModel, BertTokenizer, AutoImageProcessor, VideoMAEModel

# Visualization and progress tracking
from datasets import load_dataset
import av # pip install av

# Additional utility for iterating over combinations
import pandas as pd
from configs import CFG
from text_image import OneEncoder as TextImageEncoder


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

class AlignmentLayer(nn.Module):
    def __init__(self, input_dim=768, projection_dim=CFG.projection_dim, dropout_rate=CFG.dropout_rate, *args, **kwargs):

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


class VideoEncoder(nn.Module):
    def __init__(self, model_name=CFG.video_name, projection_dim=CFG.projection_dim,
                 trainable=False, dropout_rate=CFG.dropout_rate, *args, **kwargs):

        super(VideoEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        # Models
        self.pretrained_encoder = VideoMAEModel.from_pretrained(self.model_name)
        self.alignment_layer = AlignmentLayer(
                                              input_dim=self.pretrained_encoder.config.hidden_size,
                                              projection_dim=self.projection_dim,
                                              dropout_rate=self.dropout_rate)
        # Freeze VideoMAE
        for parameter in self.pretrained_encoder.parameters():
            parameter.requires_grad = self.trainable

    def forward(self, inputs):

        x = self.pretrained_encoder(inputs).last_hidden_state
        x = self.alignment_layer(x)
        return x

    def __call__(self, inputs):
        return self.forward(inputs)


class ModalityTokenEncoder(nn.Module):
    def __init__(self, projection_dim=CFG.projection_dim, token_size=CFG.token_size, device='cpu', *args, **kwargs):
        super(ModalityTokenEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.projection_dim = projection_dim
        self.device = device
        self.token_size = token_size
        # Models
        video_variance = torch.rand(1) * 0.5 + 0.1
        self.video_token = nn.Parameter(torch.normal(mean=0, std=video_variance.item(),
                                                      size=(self.token_size, self.projection_dim)).to(self.device))

    def forward(self):
        return self.video_token

    def __call__(self):
        return self.forward()


class OneEncoder(nn.Module):
    def __init__(self, device='cpu', modality_token_encoder=ModalityTokenEncoder(), checkpoint="bilalfaye/OneEncoder-text-image",
                 video_processor=AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base"),
                 video_encoder=VideoEncoder(), *args, **kwargs):
        super(OneEncoder, self).__init__(*args, **kwargs)

        self.device = device
        self.checkpoint = checkpoint
        self.modality_token_encoder = modality_token_encoder
        self.modality_token_encoder.device = self.device
        self.text_image_encoder = TextImageEncoder(device=self.device)
        self.text_image_encoder.from_pretrained(self.checkpoint)
        self.video_processor = video_processor
        self.video_encoder = video_encoder
        self.temperature = nn.Parameter(torch.tensor(0.07).to(self.device))

        # Freeze
        for parameter in self.text_image_encoder.parameters():
            parameter.requires_grad = False

    @classmethod
    def load_video(cls, video_path):
        container = av.open(video_path)
        return container

    @classmethod
    def read_video_pyav(cls, container, indices):
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

    @classmethod
    def sample_frame_indices(cls, clip_len, frame_sample_rate, seg_len):
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

    def encode_video(self, videos):
        """
        :param videos: torch.Size([batch, 16, 3, 224, 224])
        :return: torch.Size([batch, 1568, 768])
        """
        video_features = self.video_encoder(videos.to(self.device))
        modality_token_features = self.modality_token_encoder()
        outputs = self.text_image_encoder.universal_projection_encoder([video_features, modality_token_features]).last_hidden_state

        return outputs

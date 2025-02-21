# PyTorch for deep learning operations
import torch
import torch.nn as nn


# PyTorch data loading and utilities
import torch.multiprocessing

import torchaudio
from transformers import AutoProcessor, Wav2Vec2Model
import torchaudio.transforms as transforms
from huggingface_hub import PyTorchModelHubMixin
from configs import CFG
from text_image import OneEncoder as TextImageEncoder
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio

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

class AudioEncoder(nn.Module):
    def __init__(self, model_name=CFG.audio_name, projection_dim=CFG.projection_dim,
                 trainable=False, dropout_rate=CFG.dropout_rate, *args, **kwargs):

        super(AudioEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        # Models
        self.pretrained_encoder = Wav2Vec2Model.from_pretrained(self.model_name)
        self.alignment_layer = AlignmentLayer(
                                              input_dim=self.pretrained_encoder.config.hidden_size,
                                              projection_dim=self.projection_dim,
                                              dropout_rate=self.dropout_rate)
        # Freeze Wav2VecModel
        for parameter in self.pretrained_encoder.parameters():
            parameter.requires_grad = self.trainable
        # Unfreeze not initialized layers
        newly_initialized_layers = [
            'encoder.pos_conv_embed.conv.parametrizations.weight.original0',
            'encoder.pos_conv_embed.conv.parametrizations.weight.original1',
            'masked_spec_embed'
        ]
        for name, param in self.pretrained_encoder.named_parameters():
            if any(layer_name in name for layer_name in newly_initialized_layers):
                param.requires_grad = True

    def forward(self, inputs):

        x = self.pretrained_encoder(inputs['input_values'].float()).last_hidden_state
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
        audio_variance = torch.rand(1) * 0.5 + 0.1
        self.audio_token = nn.Parameter(torch.normal(mean=0, std=audio_variance.item(),
                                                      size=(self.token_size, self.projection_dim)).to(self.device))
    def forward(self):
        return self.audio_token

    def __call__(self):
        return self.forward()

class OneEncoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, device='cpu', modality_token_encoder=ModalityTokenEncoder(), checkpoint="bilalfaye/OneEncoder-text-image",
                 audio_processor=AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h"),
                 sample_rate=CFG.sample_rate, audio_encoder=AudioEncoder(), *args, **kwargs):
        super(OneEncoder, self).__init__(*args, **kwargs)

        self.device = device
        self.checkpoint = checkpoint
        self.modality_token_encoder = modality_token_encoder
        self.modality_token_encoder.device = self.device
        self.text_image_encoder = TextImageEncoder(device=self.device)
        self.text_image_encoder.from_pretrained(self.checkpoint)
        self.audio_processor = audio_processor
        self.sample_rate = sample_rate
        self.audio_encoder = audio_encoder
        self.temperature = nn.Parameter(torch.tensor(0.07).to(self.device))

        # Freeze
        for parameter in self.text_image_encoder.parameters():
            parameter.requires_grad = False

    def load_audio(self, audio_path):
        waveform, original_sample_rate = torchaudio.load(audio_path)
        # If the audio needs to be resampled
        if original_sample_rate != self.sample_rate:
            resampler = transforms.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # mono sound -> output shape: torch.Size(1, dim)
        # Stereo sound -> output shape: torch.Size(2, dim)
        # Surround sound -> output shape: torch.Size(n, dim)
        return waveform

    def process_audio(self, audios):
        # audios: list of numpy array
        x = self.audio_processor(audios, sampling_rate=self.sample_rate, return_tensors="pt", padding=True, max_length=15*self.sample_rate, truncation=True)
        #x = self.audio_processor(audios, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        return x

    def encode_audio(self, audios):
        # audios: torch 2D (batch, dim)
        audio_embeddings = self.audio_encoder(audios.to(self.device))
        modality_token = self.modality_token_encoder()
        audio_features = self.text_image_encoder.universal_projection_encoder([audio_embeddings, modality_token]).last_hidden_state
        return audio_features.float()

    def matching_image_audio(self, audios, image_paths=None, image_tensors=None,
                              normalize=True, top_k=None, strategy="similarity", temperature=0.0):
        # audios is of shape {"input_values":torch.Size([N, dim])}
        wav_features = torch.mean(self.encode_audio(audios), dim=1)
        image_features = self.text_image_encoder.encode_image(image_paths=image_paths, image_tensors=image_tensors)
        if normalize:
            image_features = F.normalize(image_features, p=2, dim=-1)
            wav_features = F.normalize(wav_features, p=2, dim=-1)
        dot_similarities = (image_features @ wav_features.T) * torch.exp(torch.tensor(temperature).to(self.device))
        if strategy == 'softmax':
            dot_similarities = (float(audios["input_values"].shape[0]) * dot_similarities).softmax(dim=-1)
        if top_k is not None:
            top_probs, top_labels = dot_similarities.cpu().topk(top_k, dim=-1)
            return top_probs, top_labels
        else:
            return dot_similarities, None

    def matching_text_audio(self, audios, texts, normalize=True, top_k=None, strategy="similarity", temperature=0.0):
        # audios is of shape {"input_values":torch.Size([N, dim])}
        wav_features = torch.mean(self.encode_audio(audios), dim=1)
        text_features = self.text_image_encoder.encode_text(texts=texts)
        if normalize:
            text_features = F.normalize(text_features, p=2, dim=-1)
            wav_features = F.normalize(wav_features, p=2, dim=-1)
        dot_similarities = (text_features @ wav_features.T) * torch.exp(torch.tensor(temperature).to(self.device))
        if strategy == 'softmax':
            dot_similarities = (float(audios["input_values"].shape[0]) * dot_similarities).softmax(dim=-1)
        if top_k is not None:
            top_probs, top_labels = dot_similarities.cpu().topk(top_k, dim=-1)
            return top_probs, top_labels
        else:
            return dot_similarities, None

    def image_retrieval(self, query, image_paths, image_embeddings=None, temperature=0.0, n=9, plot=False, display_audio=False):
        # query is of shape {"input_values":torch.Size([1, dim])}
        wav_embeddings = torch.mean(self.encode_audio(audios=query), dim=1)
        if image_embeddings is None:
            image_embeddings = self.text_image_encoder.encode_image(image_paths=image_paths)

        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        wav_embeddings_n = F.normalize(wav_embeddings, p=2, dim=-1)
        dot_similarity = (wav_embeddings_n @ image_embeddings_n.T) * torch.exp(
            torch.tensor(temperature).to(self.device))
        if n > len(image_paths):
            n = len(image_paths)
        values, indices = torch.topk(dot_similarity.cpu().squeeze(0), n)
        if plot:
            nrows = int(np.sqrt(n))
            ncols = int(np.ceil(n / nrows))
            matches = [image_paths[idx] for idx in indices]
            fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
            for match, ax in zip(matches, axes.flatten()):
                image = self.text_image_encoder.load_image(f"{match}")
                ax.imshow(image)
                ax.axis("off")
            plt.savefig("img.png")
            if display_audio:
                fig.suptitle(display(Audio(query['input_values'], rate=self.sample_rate)))
            plt.show()
        return values, indices
       

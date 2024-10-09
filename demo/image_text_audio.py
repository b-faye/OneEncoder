from image_text import OneEncoderUP
from configs import CFG
import torch.nn as nn
import torch
from transformers import AutoProcessor, Wav2Vec2Model, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch.nn.functional as F
from tqdm import tqdm
import os
from costum_datasets import make_pairs, build_loaders, download_dataset
import random
import itertools
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.display import Audio

################################################### TRAINING STEP ######################################################
################################################# TRAINING STEP ########################################################


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=CFG.projection_dim, projection_dim=CFG.projection_dim, dropout_rate=CFG.dropout_rate, *args, **kwargs):

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


class WavEncoder(nn.Module):
    def __init__(self, model_name=CFG.wav_name, projection_dim=CFG.projection_dim,
                 trainable=False, dropout_rate=CFG.dropout_rate, *args, **kwargs):

        super(WavEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        # Models
        self.pretrained_wav = Wav2Vec2Model.from_pretrained(self.model_name)
        self.projection_head = ProjectionHead(self.pretrained_wav.config.hidden_size,
                                              self.projection_dim, self.dropout_rate)
        # Freeze Wav2VecModel
        for parameter in self.pretrained_wav.parameters():
            parameter.requires_grad = self.trainable
        # Unfreeze not initialized layers
        newly_initialized_layers = [
            'encoder.pos_conv_embed.conv.parametrizations.weight.original0',
            'encoder.pos_conv_embed.conv.parametrizations.weight.original1',
            'masked_spec_embed'
        ]
        for name, param in self.pretrained_wav.named_parameters():
            if any(layer_name in name for layer_name in newly_initialized_layers):
                param.requires_grad = True

    def forward(self, inputs):

        x = self.pretrained_wav(inputs).last_hidden_state
        x = self.projection_head(x)
        return x

    def __call__(self, inputs):
        return self.forward(inputs)


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
        wav_variance = torch.rand(1) * 0.5 + 0.1
        self.wav_context = nn.Parameter(torch.normal(mean=0, std=wav_variance.item(),
                                                     size=(1, self.context_input_dim)).to(self.device))
        self.model = nn.Sequential(
            nn.Linear(self.context_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        )

    def forward(self):
        return self.model(self.wav_context)

    def __call__(self):
        return self.forward()


################################################### TRAINING STEP ######################################################
################################################# TRAINING STEP ########################################################

class OneEncoderAL(nn.Module):
    def __init__(self, pretrained_weights="best.pt", alignment_encoder=ProjectionHead(),
                 device=CFG.device, wav_encoder=WavEncoder(), context_encoder=ContextEncoder(),
                 frugcrl_trainable=False, *args, **kwargs):
        super(OneEncoderAL, self).__init__(*args, **kwargs)
        self.device = device
        self.to(self.device)
        self.pretrained_frugcrl = OneEncoderUP(device=self.device)
        self.pretrained_frugcrl.load_state_dict(torch.load(pretrained_weights, map_location=self.device))
        self.alignment_encoder = alignment_encoder.to(self.device)
        self.frugcrl_trainable = frugcrl_trainable
        self.wav_encoder = wav_encoder.to(self.device)
        self.context_encoder = context_encoder.to(self.device)

        #self.text_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        #self.audio_generator = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        #self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        # load xvector containing speaker's voice characteristics from a dataset
        #embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        #self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        self.temperature = nn.Parameter(torch.tensor(0.07).to(self.device))
        for parameter in self.pretrained_frugcrl.parameters():
            parameter.requires_grad = self.frugcrl_trainable

    def encode_audio(self, audios, wav_processor=AutoProcessor.from_pretrained(CFG.wav_name),
                     outputs="average", duration_seconds=10):
        processed_input = []
        list_size = len(audios)
        for audio in audios:
            if not audio.lower().endswith('.wav'):
                raise TypeError("Audio must be in wav format")
            data, sample_rate = sf.read(audio)

            desired_length = duration_seconds * sample_rate
            current_length = data.shape[0]
            if current_length < desired_length:
                # Pad the audio data with zeros
                padding = torch.zeros((desired_length - current_length,))
                data_padded = torch.cat((torch.tensor(data), padding))
            elif current_length > desired_length:
                # Truncate the audio data
                data_padded = data[:desired_length]
            else:
                # No need for padding or truncation
                data_padded = data
            tensor_audio = wav_processor(data_padded, sampling_rate=16000, return_tensors="pt")
            processed_input.append(tensor_audio["input_values"].squeeze(0))

        processed_input = torch.stack(processed_input)
        with torch.no_grad():
            wav_features = self.wav_encoder(processed_input.to(self.device))
            wav_context_feature = self.context_encoder()
            wav_context_features = wav_context_feature.repeat(list_size, 1)
            output_features = self.pretrained_frugcrl.fusion_encoder([wav_features, wav_context_features])
            if outputs == "average":
                wav_features = output_features.average_outputs
            elif outputs == "min":
                wav_features = output_features.min_outputs
            elif outputs == "max":
                wav_features = output_features.max_outputs
            else:
                wav_features = output_features.sequence_outputs
        return wav_features

    def matching_image_audio(self, audios, image_paths=None, image_tensors=None,
                             normalize=True, top_k=None, strategy="similarity", temperature=2.5):
        wav_features = self.encode_audio(audios)
        image_features = self.pretrained_frugcrl.encode_image(image_paths=image_paths, image_tensors=image_tensors)
        if normalize:
            image_features = F.normalize(image_features, p=2, dim=-1)
            wav_features = F.normalize(wav_features, p=2, dim=-1)
        dot_similarities = (image_features @ wav_features.T) * torch.exp(torch.tensor(temperature).to(self.device))
        if strategy == 'softmax':
            dot_similarities = (float(len(audios)) * dot_similarities).softmax(dim=-1)
        if top_k is not None:
            top_probs, top_labels = dot_similarities.cpu().topk(top_k, dim=-1)
            return top_probs, top_labels
        else:
            return dot_similarities, None

    def matching_text_audio(self, audios, texts, normalize=True, top_k=None, strategy="similarity", temperature=2.5):
        wav_features = self.encode_audio(audios)
        text_features = self.pretrained_frugcrl.encode_text(texts=texts)
        if normalize:
            text_features = F.normalize(text_features, p=2, dim=-1)
            wav_features = F.normalize(wav_features, p=2, dim=-1)
        dot_similarities = (text_features @ wav_features.T) * torch.exp(torch.tensor(temperature).to(self.device))
        if strategy == 'softmax':
            dot_similarities = (float(len(audios)) * dot_similarities).softmax(dim=-1)
        if top_k is not None:
            top_probs, top_labels = dot_similarities.cpu().topk(top_k, dim=-1)
            return top_probs, top_labels
        else:
            return dot_similarities, None

    def image_retrieval(self, query, image_paths, image_embeddings=None, temperature=2.5, n=9, plot=False):

        wav_embeddings = self.encode_audio(audios=[query])
        if image_embeddings is None:
            image_embeddings = self.pretrained_frugcrl.encode_image(image_paths=image_paths)

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
            fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
            for match, ax in zip(matches, axes.flatten()):
                image = self.pretrained_frugcrl.load_image(f"{match}")
                ax.imshow(image)
                ax.axis("off")

            fig.suptitle(display(Audio(query)))
            plt.show()
        return values, indices

    def forward(self, inputs):

        images = inputs["image"]
        audios = inputs["audio"]

        image_features = self.pretrained_frugcrl.encode_image(image_tensors=images)
        wav_features = self.wav_encoder(audios)
        wav_context_feature = self.context_encoder()
        wav_context_features = wav_context_feature.repeat(audios.size(0), 1)
        wav_features = self.pretrained_frugcrl.fusion_encoder([wav_features, wav_context_features]).average_outputs

        # For constant temperature replace self.temperature by self.temperature.data and give the target temperature
        # for example temperature = 1.0 or temperature = 2.5
        t = torch.clamp(self.temperature.data, min=torch.tensor(2.5).to(self.device),
                        max=torch.tensor(3.0).to(self.device))

        # L2 normalization
        image_features = F.normalize(image_features, p=2, dim=-1)
        wav_features = F.normalize(wav_features, p=2, dim=-1)

        # Compute loss
        logits = (image_features @ wav_features.T) * torch.exp(t)
        # labels
        labels = torch.arange(logits.size(0)).to(self.device)
        loss_i = F.cross_entropy(logits, labels, reduction='mean')
        loss_w = F.cross_entropy(logits.t(), labels, reduction='mean')
        loss = (loss_i + loss_w) / 2.0

        return loss

    def call(self, inputs):
        return self.forward(inputs)

################################################### TRAINING STEP ######################################################
################################################# TRAINING STEP ########################################################

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

        batch = {k: v.to(CFG.device) for k, v in batch.items()}
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
        batch = {k: v.to(CFG.device) for k, v in batch.items()}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


################################################### TRAINING STEP ######################################################
################################################# TRAINING STEP ########################################################


def main(basic_train=True, pretrained_frugcrl_weights="frugcrl.pt"):
    # Load the pretrained Wav2Vec2Model processor
    wav_processor = AutoProcessor.from_pretrained(CFG.wav_name)

    # Load COCO dataset if not exist
    if len(os.listdir(CFG.data_directory)) == 0:
        download_dataset()
    # Create pairs image-caption 413.915
    training_pairs = make_pairs(CFG.train_annotation_file, CFG.image_dir, "../datasets/coco_audio_train",
                                5)
    random.shuffle(training_pairs)
    # validation 202.520
    validation_pairs = make_pairs(CFG.val_annotation_file, CFG.image_dir_val, "../datasets/coco_audio_val",
                                  5)
    random.shuffle(validation_pairs)
    validation_pairs = validation_pairs[-round(len(validation_pairs)*0.20):]
    print("Number of training images: {}".format(len(training_pairs)))
    print("Number of validation images: {}".format(len(validation_pairs)))
    # Build loader : return dictionary
    train_loader = build_loaders(training_pairs, audio_processor=wav_processor, mode="train")
    val_loader = build_loaders(validation_pairs, audio_processor=wav_processor, mode="valid")
    # Create the training model
    model = HierarchicFrugCRL(pretrained_weights=pretrained_frugcrl_weights, device=CFG.device)
    if basic_train:
        # Train all parameters with the same lr and weight decay
        # This method is better when using dynamique temperature
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=CFG.weight_decay, lr=CFG.lr)
    else:
        parameters = [
            {"params": model.pretrained_frugcrl.parameters(), "lr": CFG.wav_encoder_lr},
            {"params": model.wav_encoder.pretrained_wav.parameters(), "lr": CFG.wav_encoder_lr},
            {
                "params": itertools.chain(
                    model.context_encoder.parameters(), model.wav_encoder.projection_head.parameters(),
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
            print("Saved best model!")
        else:
            if epoch >= CFG.patience - 1:
                num_bad_epochs += 1
            if num_bad_epochs >= CFG.patience:
                print(f"Early stopping at epoch {epoch + 1}. Restoring best weights...")
                break
        lr_scheduler.step(val_loss.avg)
    torch.save(model.state_dict(), "last.pt")
    # Free GPU
    model = None
    optimizer = None
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


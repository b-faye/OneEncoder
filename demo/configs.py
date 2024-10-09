import torch
import os


class CFG:
    sample_rate = 16000
    duration_seconds = 10
    batch_size = 64
    num_workers = 4
    projection_dim = 256
    dropout_rate = 0.1
    context_input_dim = 3
    num_head = 4
    num_layers = 1
    image_encoder_lr = 1e-4
    wav_encoder_lr = 1e-5
    context_encoder_lr = 1e-3
    fusion_lr = 1e-3
    lr = 1e-3
    weight_decay = 1e-3
    patience = 5
    factor = 0.8
    epochs = 100
    image_size = 224
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_directory = "../datasets"
    train_annotation_file = os.path.join(data_directory, "annotations", "captions_train2014.json")
    val_annotation_file = os.path.join(data_directory, "annotations", "captions_val2014.json")
    image_dir = os.path.join(data_directory, "train2014")
    image_dir_val = os.path.join(data_directory, "val2014")
    wav_name = "facebook/wav2vec2-base-960h"
    vit_name = "vit_base_patch16_224"
    bert_name = "bert-base-uncased"

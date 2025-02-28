import torch
import os

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


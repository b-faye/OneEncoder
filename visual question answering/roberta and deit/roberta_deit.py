# Done
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, set_caching_enabled
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import (
    # Preprocessing / Common
    AutoTokenizer, AutoFeatureExtractor,
    # Text and Image models
    AutoModel,
    # Training / Evaluation
    Trainer, TrainingArguments,
    # Misc
    logging
)
import nltk

nltk.download('wordnet')
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score, f1_score
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
#import enchant
from numpy import prod
from nltk.corpus import wordnet as wn

# SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
#os.environ['HF_HOME'] = os.path.join(".", "cache")
# SET ONLY 1 GPU DEVICE
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#set_caching_enabled(True)
#logging.set_verbosity_error()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_daquar_dataset(dataset_path="../../datasets/dataset"):
    # Define a regex pattern to normalize the question &
    # find the image ID for which the question is asked
    image_pattern = re.compile("( (in |on |of )?(the |this )?(image\d*) \?)")

    with open(os.path.join(dataset_path, "all_qa_pairs.txt")) as f:
        qa_data = [x.replace("\n", "") for x in f.readlines()]

    records = []
    for i in range(0, len(qa_data), 2):
        img_id = image_pattern.findall(qa_data[i])[0][3]
        question = qa_data[i].replace(image_pattern.findall(qa_data[i])[0][0], "")
        record = {
            "question": question,
            "answer": qa_data[i + 1],
            "image_id": img_id,
        }
        records.append(record)

    df = pd.DataFrame(records)
    # Create a list of all possible answers, so that the answer generation part of the VQA task
    # can be modelled as multiclass classification
    answer_space = []
    for ans in df.answer.to_list():
        answer_space = answer_space + [ans] if "," not in ans else answer_space + ans.replace(" ", "").split(",")

    answer_space = list(set(answer_space))
    answer_space.sort()
    with open(os.path.join(dataset_path, "answer_space.txt"), "w") as f:
        f.writelines("\n".join(answer_space))

    # Since the actual dataset contains only ~54% of the data for training (very less),
    # we produce our own splits for training & evaluation with 80% data being used for training
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(os.path.join(dataset_path, "data_train.csv"), index=None)
    test_df.to_csv(os.path.join(dataset_path, "data_eval.csv"), index=None)

    # Load the training & evaluation dataset present in CSV format
    dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(dataset_path, "data_train.csv"),
            "test": os.path.join(dataset_path, "data_eval.csv")
        }
    )

    # Load the space of all possible answers
    with open(os.path.join(dataset_path, "answer_space.txt")) as f:
        answer_space = f.read().splitlines()

    # Since we model the VQA task as a multiclass classification problem,
    # we need to create the labels from the actual answers
    dataset = dataset.map(
        lambda examples: {
            'label': [
                # Select the 1st answer if multiple answers are provided for single question
                answer_space.index(ans.replace(" ", "").split(",")[0])
                for ans in examples['answer']
            ]
        },
        batched=True
    )

    return dataset, answer_space

dataset, answer_space = load_daquar_dataset()

# The dataclass decorator is used to automatically generate special methods to classes,
# including __init__, __str__ and __repr__. It helps reduce some boilerplate code.
@dataclass
class MultimodalCollator:
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor

    def tokenize_text(self, texts: List[str]):
        encoded_text = self.tokenizer(
            text=texts,
            padding='longest',
            max_length=24,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
        }

    def preprocess_images(self, images: List[str]):
        processed_images = self.preprocessor(
            images=[Image.open(os.path.join("../../datasets/dataset", "image", image_id + ".png")).convert('RGB')
                    for image_id in images],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }

    def __call__(self, raw_batch_dict):
        return {
            **self.tokenize_text(
                raw_batch_dict['question']
                if isinstance(raw_batch_dict, dict) else
                [i['question'] for i in raw_batch_dict]
            ),
            **self.preprocess_images(
                raw_batch_dict['image_id']
                if isinstance(raw_batch_dict, dict) else
                [i['image_id'] for i in raw_batch_dict]
            ),
            'labels': torch.tensor(
                raw_batch_dict['label']
                if isinstance(raw_batch_dict, dict) else
                [i['label'] for i in raw_batch_dict],
                dtype=torch.int64
            ),
        }


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=768, dropout_rate=0.2, *args, **kwargs):

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


class ModalityTokenEncoder(nn.Module):
    def __init__(self, projection_dim=768, token_size=1, device='cpu', *args, **kwargs):

        super(ModalityTokenEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.projection_dim = projection_dim
        self.device = device
        self.token_size = token_size
        # Models
        text_variance = torch.rand(1) * 0.5 + 0.1
        image_variance = torch.rand(1) * 0.5 + 0.1
        self.text_token = nn.Parameter(torch.normal(mean=0, std=text_variance.item(),
                                                      size=(self.token_size, self.projection_dim)).to(self.device))
        self.image_token = nn.Parameter(torch.normal(mean=0, std=image_variance.item(),
                                                       size=(self.token_size, self.projection_dim)).to(self.device))

    def forward(self, modality_type):

        token = torch.where(torch.tensor(modality_type == "image"), self.image_token, self.text_token)
        return token

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
    def __init__(self, input_dim=768, num_head=4, num_layers=2, *args, **kwargs):

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
        self.layer_normalization = nn.LayerNorm(self.input_dim)

    def forward(self, inputs):
        # x: image or caption embeddings
        x, tokens = inputs

        ## Universal Projection block
        tokens = tokens.unsqueeze(0).expand(x.size()[0], -1, -1)

        # Concatenate tokens with image/caption embeddings
        output_tensor = torch.cat((tokens, x), dim=1)
        output_tensor = x + tokens
        
        # Norm
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



class MultimodalVQAModel(nn.Module):
    def __init__(self, pretrained_text_name, pretrained_image_name, num_labels=len(answer_space), intermediate_dim=512,
                 dropout=0.5, hidden_size=768, trainable=False):
        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.trainable = trainable

        # Pretrained transformers for text & image featurization
        self.text_encoder = AutoModel.from_pretrained(self.pretrained_text_name)
        self.image_encoder = AutoModel.from_pretrained(self.pretrained_image_name)

        # projection
        self.text_projection = ProjectionHead(input_dim=self.text_encoder.config.hidden_size,
                                              projection_dim=self.hidden_size)
        self.image_projection = ProjectionHead(input_dim=self.image_encoder.config.hidden_size,
                                               projection_dim=self.hidden_size)

        # Modality token encoder
        self.modality_token_encoder = ModalityTokenEncoder()

        # Universal Projection Encoder
        self.universal_projection_encoder = UniversalProjectionEncoder(input_dim=self.hidden_size)

        # Fusion layer for cross-modal interaction
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size + self.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # Fully-connected classifier
        self.classifier = nn.Linear(intermediate_dim, self.num_labels)

        self.criterion = nn.CrossEntropyLoss()
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = self.trainable

        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = self.trainable

    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None):
        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_text = self.text_projection(encoded_text.last_hidden_state)
        encoded_text = self.universal_projection_encoder([encoded_text, self.modality_token_encoder("text")])
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        encoded_image = self.image_projection(encoded_image.last_hidden_state)
        encoded_image = self.universal_projection_encoder([encoded_image, self.modality_token_encoder("image")])
        fused_output = self.fusion(
            torch.cat(
                [
                    encoded_text.mean_output,
                    encoded_image.mean_output,
                ],
                dim=1
            )
        )
        logits = self.classifier(fused_output)

        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss

        return out


def createMultimodalVQACollatorAndModel(text='bert-base-uncased', image='google/vit-base-patch16-224-in21k'):
    # Initialize the correct text tokenizer and image feature extractor, and use them to create the collator
    tokenizer = AutoTokenizer.from_pretrained(text)
    preprocessor = AutoFeatureExtractor.from_pretrained(image)
    multimodal_collator = MultimodalCollator(tokenizer=tokenizer, preprocessor=preprocessor)

    # Initialize the multimodal model with the appropriate weights from pretrained models
    multimodal_model = MultimodalVQAModel(pretrained_text_name=text, pretrained_image_name=image).to(device)

    return multimodal_collator, multimodal_model




def file2list(filepath):
    with open(filepath,'r') as f:
        lines =[k for k in
            [k.strip() for k in f.readlines()]
        if len(k) > 0]

    return lines


def list2file(filepath,mylist):
    mylist='\n'.join(mylist)
    with open(filepath,'w') as f:
        f.writelines(mylist)


def items2list(x):
    """
    x - string of comma-separated answer items
    """
    return [l.strip() for l in x.split(',')]


def fuzzy_set_membership_measure(x,A,m):
    """
    Set membership measure.
    x: element
    A: set of elements
    m: point-wise element-to-element measure m(a,b) ~ similarity(a,b)

    This function implments a fuzzy set membership measure:
        m(x \in A) = max_{a \in A} m(x,a)}
    """
    return 0 if A==[] else max(map(lambda a: m(x,a), A))


def score_it(A,T,m):
    """
    A: list of A items
    T: list of T items
    m: set membership measure
        m(a \in A) gives a membership quality of a into A

    This function implements a fuzzy accuracy score:
        score(A,T) = min{prod_{a \in A} m(a \in T), prod_{t \in T} m(a \in A)}
        where A and T are set representations of the answers
        and m is a measure
    """
    if A==[] and T==[]:
        return 1

    # print A,T

    score_left=0 if A==[] else prod(map(lambda a: m(a,T), A))
    score_right=0 if T==[] else prod(map(lambda t: m(t,A),T))
    return min(score_left,score_right)


# implementations of different measure functions
def dirac_measure(a,b):
    """
    Returns 1 iff a=b and 0 otherwise.
    """
    if a==[] or b==[]:
        return 0.0
    return float(a==b)


def wup_measure(a,b,similarity_threshold=0.925):
    """
    Returns Wu-Palmer similarity score.
    More specifically, it computes:
        max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
        where interp is a 'interpretation field'
    """
    def get_semantic_field(a):
        weight = 1.0
        semantic_field = wn.synsets(a,pos=wn.NOUN)
        return (semantic_field,weight)


    def get_stem_word(a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        return (a,weight)


    global_weight=1.0

    (a,global_weight_a)=get_stem_word(a)
    (b,global_weight_b)=get_stem_word(b)
    global_weight = min(global_weight_a,global_weight_b)

    if a==b:
        # they are the same
        return 1.0*global_weight

    if a==[] or b==[]:
        return 0


    interp_a,weight_a = get_semantic_field(a)
    interp_b,weight_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0

    # we take the most optimistic interpretation
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if local_score > global_max:
                global_max=local_score

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score=global_max*weight_a*weight_b*interp_weight*global_weight
    return final_score


# Wrapper around the wup_measure(...) function to process batch inputs
def batch_wup_measure(labels, preds):
    wup_scores = [wup_measure(answer_space[label], answer_space[pred]) for label, pred in zip(labels, preds)]
    return np.mean(wup_scores)


# Function to compute all relevant performance metrics, to be passed into the trainer
def compute_metrics(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_tuple
    preds = logits.argmax(axis=-1)
    return {
        "wups": batch_wup_measure(labels, preds),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro')
    }


def main():
    multi_args = TrainingArguments(
        output_dir="checkpoint",
        seed=12345,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        # Since models are large, save only the last 3 checkpoints at any given time while training
        metric_for_best_model='wups',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        remove_unused_columns=False,
        num_train_epochs=100,
        fp16=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
    )

    # Initialize the actual collator and multimodal model
    collator, model = createMultimodalVQACollatorAndModel("roberta-base",
                                                          "facebook/deit-base-distilled-patch16-224")

    # Initialize the trainer with the dataset, collator, model, hyperparameters and evaluation metrics
    multi_trainer = Trainer(
        model,
        multi_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    # Start the training loop
    train_multi_metrics = multi_trainer.train()
    print("############################# TRAINING #######################################")
    print(train_multi_metrics)

    # Run the model on the evaluation set to obtain final metrics
    print("############################# EVALUATION #######################################")
    eval_multi_metrics = multi_trainer.evaluate()

    print(eval_multi_metrics)


if __name__ == "__main__":
    main()

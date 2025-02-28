import torch
import torchaudio.transforms as T
import numpy as np
from datasets import load_dataset
from sklearn.metrics import average_precision_score
import torch.nn.functional as F

from audio_image import OneEncoder as AudioImageEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

urban = load_dataset("danavery/urbansound8K", trust_remote_code=True)
urban['train'] = urban['train'].map(lambda x: {'class': x['class'].replace('_', ' ')})
class_names = list(set(urban['train']['class']))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset['train']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio_array = self.dataset[idx]['audio']['array']
        audio_sample_rate = self.dataset[idx]['audio']['sampling_rate']
        label = self.dataset[idx]['class']
        resampler = T.Resample(audio_sample_rate, 16000)
        audio_tensor = resampler(torch.tensor(audio_array, dtype=torch.float32))
        audio_array = audio_tensor.squeeze().numpy()

        return audio_array, label


def collate_fn(batch):
    audio_list, label_list = zip(*batch)
    return list(audio_list), list(label_list)


train_dataset = Dataset(urban)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)

model = AudioImageEncoder(device=device)
model.load_state_dict(torch.load("audio_image.pt", map_location=device))
model = model.to(device)
model.eval()

label_features = model.text_image_encoder.encode_text(texts=class_names)
label_feature = F.normalize(label_features, p=2, dim=-1)

similarities = {name: [] for name in class_names}
all_labels = {name: [] for name in class_names}

for audios, labels in train_dataloader:
    audios_processed = model.process_audio(audios)
    with torch.no_grad():
        audio_features = torch.mean(model.encode_audio(audios_processed), dim=1)

    audio_features = F.normalize(audio_features, p=2, dim=-1)

    similaritiy = (audio_features @ label_feature.T)
    for idx, name in enumerate(class_names):
        similarities[name].extend(similaritiy[idx].tolist())
        all_labels[name].extend(labels)


# Function to calculate P@1, R@1, and mAP
def calculate_metrics(similarities, all_labels, class_names):
    p_at_1 = 0
    r_at_1 = 0
    average_precisions = []

    for name in class_names:
        sim_scores = similarities[name]
        true_labels = all_labels[name]

        sorted_indices = np.argsort(sim_scores)[::-1]  # Indices sorted by similarity score (descending)

        # Calculate P@1 and R@1
        if true_labels[sorted_indices[0]] == name:
            p_at_1 += 1
            r_at_1 += 1

        # Calculate Average Precision for mAP
        y_true = np.array([1 if label == name else 0 for label in true_labels])
        y_scores = np.array(sim_scores)
        average_precisions.append(average_precision_score(y_true, y_scores))

    p_at_1 /= len(class_names)
    r_at_1 /= len(class_names)
    mAP = np.mean(average_precisions)

    return p_at_1, r_at_1, mAP


# Calculate and print the metrics
p_at_1, r_at_1, mAP = calculate_metrics(similarities, all_labels, class_names)
print(f"P@1: {p_at_1}")
print(f"R@1: {r_at_1}")
print(f"mAP: {mAP}")

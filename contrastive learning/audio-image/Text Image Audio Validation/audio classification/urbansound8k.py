import torch
import torchaudio.transforms as T
import numpy as np
from datasets import load_dataset
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from audio_image import OneEncoder as AudioImageEncoder


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio_array = self.dataset[idx]['audio']['array']
        audio_sample_rate = self.dataset[idx]['audio']['sampling_rate']
        label = self.dataset[idx]['classID']
        resampler = T.Resample(audio_sample_rate, 16000)
        audio_tensor = resampler(torch.tensor(audio_array, dtype=torch.float32))
        audio_array = audio_tensor.squeeze().numpy()

        return audio_array, label


def collate_fn(batch):
    audio_list, label_list = zip(*batch)
    return list(audio_list), torch.tensor(label_list, dtype=torch.int32)


class LinearClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes=10, trainable=False, device='cpu', *args, **kwargs):
        super(LinearClassifier, self).__init__(*args, **kwargs)
        self.model = pretrained_model
        self.num_classes = num_classes
        self.trainable = trainable
        self.device = device
        self.classifier = nn.Sequential(
            nn.Linear(self.model.text_image_encoder.universal_projection_encoder.input_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),

            nn.Linear(32, num_classes)
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = self.trainable
        self.to(self.device)
        self.model.to(self.device)
        self.classifier.to(self.device)

    def forward(self, inputs):
        audio_features = torch.mean(self.model.encode_audio(audios=inputs), dim=1)
        logits = self.classifier(audio_features)
        return logits

    def accuracy(self, data_loader):
        top_accuracy = {f"top_{i+1}_accuracy": 0 for i in range(5)}
        total_samples = 0
        with torch.no_grad():
            self.eval()
            for inputs, labels in tqdm(data_loader, desc='Validation'):
                labels = labels.to(self.device)
                inputs = self.model.process_audio(inputs)
                inputs = inputs.to(self.device)
                total_samples += labels.size(0)
                logits = self(inputs)
                _, predicted_top_k = torch.topk(logits, 5, dim=1)
                for i in range(5):
                    top_accuracy[f"top_{i+1}_accuracy"] += torch.sum(torch.any(
                        predicted_top_k[:, :i+1] == labels.view(-1, 1), dim=-1)).item()

        for name in top_accuracy:
            top_accuracy[name] /= total_samples

        return top_accuracy

    def __call__(self, inputs):
        return self.forward(inputs)


def main(model_path="audio_image.pt", trainable=False, epochs=10):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_model = AudioImageEncoder(device=device)
    pretrained_model.load_state_dict(torch.load(model_path, map_location=device))
    urban = load_dataset("danavery/urbansound8K", trust_remote_code=True)
    urban = urban.shuffle()
    # Manually split the dataset
    total_size = len(urban['train'])
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_data = urban['train'].select(range(train_size))
    test_data = urban['train'].select(range(train_size, total_size))
    train_data = Dataset(train_data)
    test_data = Dataset(test_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)
    model = LinearClassifier(pretrained_model=pretrained_model, device=device, trainable=trainable)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            labels = labels.to(device)
            inputs = pretrained_model.audio_processor(inputs)
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        if (epoch + 1) % 10 == 0:
            # Validate the model on the test set
            top_accuracy = model.accuracy(test_loader)
            print(top_accuracy)

    # Save the fine-tuned model if needed
    torch.save(model.state_dict(), 'one_encoder_linear_classifier.pt')


if __name__ == "__main__":
    main()

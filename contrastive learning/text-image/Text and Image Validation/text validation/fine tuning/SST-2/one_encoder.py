import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import List, Dict, Tuple
from text_image import OneEncoder as TextImageEncoder
from transformers import AdamW

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[str, int]:
        text = self.data[idx]['sentence']
        label = self.data[idx]['label']
        return text, label


def collate_fn(batch) :
    texts, labels = zip(*batch)
    return texts, torch.tensor(labels)



class Classifier(nn.Module):
    def __init__(self, pretrained_model, hidden_size=CFG.projection_dim, num_labels=2):
        super(Classifier, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Freeze the XLNet layers
        for param in self.pretrained_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        outputs = self.pretrained_model.encode_text(texts=inputs)
        logits = self.classifier(outputs)
        return logits



def train_model(model, train_loader, val_loader, optimizer, device, epochs=100):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            inputs, labels = batch
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = correct / total
        if (epoch + 1) % 10 == 0:
            val_acc = evaluate_model(model, val_loader, device)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            labels = labels.to(device)
            logits = model(inputs)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total

def main():
    # Load the dataset
    sst = load_dataset("stanfordnlp/sst2")

    # Create DataLoader for each split
    train_dataset = CustomDataset(sst['train'])
    val_dataset = CustomDataset(sst['validation'])
    test_dataset = CustomDataset(sst['test'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pretrained_model = TextImageEncoder(device=device)
    pretrained_model.load_state_dict(torch.load("text_image.pt", map_location=device))

    model = Classifier(pretrained_model)
    model = model.to(device)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, device, epochs=100)

    # Evaluate the model on the test set
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import XLNetTokenizer, XLNetModel, AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from typing import List, Dict, Tuple


class TREC_Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[str, int]:
        text = self.data[idx]['text']
        label = self.data[idx]['coarse_label']
        return text, label


def collate_fn(batch: List[Tuple[str, int]], tokenizer: XLNetTokenizer, max_length: int) -> Tuple[
    Dict[str, torch.Tensor], torch.Tensor]:
    texts, labels = zip(*batch)
    encodings = tokenizer(list(texts), truncation=True, padding='max_length', max_length=max_length,
                          return_tensors='pt')

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = torch.tensor(labels, dtype=torch.long)

    return (input_ids, attention_mask), labels


class XLNetClassifier(nn.Module):
    def __init__(self, xlnet_model, hidden_size=768, num_labels=6):
        super(XLNetClassifier, self).__init__()
        self.xlnet = xlnet_model
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Freeze the XLNet layers
        for param in self.xlnet.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.xlnet(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, -1, :]  # Use the last token for classification
        logits = self.classifier(cls_output)
        return logits


def train_model(model, train_loader, test_loader, optimizer, device, epochs=100):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for (input_ids, attention_mask), labels in train_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in [input_ids, attention_mask, labels]]
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = correct / total
        if (epoch + 1) % 10 == 0:
            test_acc = evaluate_model(model, test_loader, device)
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (input_ids, attention_mask), labels in data_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in [input_ids, attention_mask, labels]]
            logits = model(input_ids, attention_mask)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total


def main():
    # Load the dataset
    trec = load_dataset("CogComp/trec", trust_remote_code=True)

    # Initialize the tokenizer
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    max_length = 128

    # Create DataLoader for each split
    train_dataset = TREC_Dataset(trec['train'], tokenizer, max_length)
    test_dataset = TREC_Dataset(trec['test'], tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length)
    )

    # Load the pre-trained XLNet model
    xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased')

    # Initialize the XLNet classifier model
    model = XLNetClassifier(xlnet_model, num_labels=6)

    # Move model to device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Train the model
    train_model(model, train_loader, test_loader, optimizer, device, epochs=100)

    # Evaluate the model on the test set
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()

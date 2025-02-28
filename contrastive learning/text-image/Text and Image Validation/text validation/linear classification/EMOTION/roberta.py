import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score

class EmotionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, label

class RoBERTaClassifier(nn.Module):
    def __init__(self, roberta_model, hidden_size=768, num_labels=6):  # Adjust num_labels for EMOTION dataset
        super(RoBERTaClassifier, self).__init__()
        self.roberta = roberta_model
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Freeze the RoBERTa layers
        for param in self.roberta.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

def train_model(model, train_loader, val_loader, optimizer, device, epochs=100):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
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
            val_acc = evaluate_model(model, val_loader, device)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            logits = model(input_ids, attention_mask)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total

def main():
    # Load the dataset
    emotion = load_dataset("dair-ai/emotion", trust_remote_code=True)

    # Initialize the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_length = 128

    # Create DataLoader for each split
    train_dataset = EmotionDataset(emotion['train'], tokenizer, max_length)
    val_dataset = EmotionDataset(emotion['validation'], tokenizer, max_length)
    test_dataset = EmotionDataset(emotion['test'], tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the pre-trained RoBERTa model
    roberta_model = RobertaModel.from_pretrained('roberta-base')

    # Initialize the RoBERTa classifier model
    model = RoBERTaClassifier(roberta_model, num_labels=6)  # Adjust num_labels for EMOTION dataset

    # Move model to device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, device, epochs=100)

    # Evaluate the model on the test set
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Decorator pour mes analyses de temps
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"[TIME] {method.__name__} executed in {(te - ts):.2f} seconds")
        return result, te - ts
    return timed

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {device}")

# Verif gpu parce que mskn sur le processeur ça prend 20 ans
if torch.cuda.is_available():
    print(f"[*] GPU Name: {torch.cuda.get_device_name(device)}")
    print(f"[*] CUDA Version: {torch.version.cuda}")
    print(f"[*] GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.2f} GB")

# Classe pour adapter au dataset
class EmailDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = self.data.iloc[index, :-1].values.astype(np.float32)
        label = self.data.iloc[index, -1]
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)

# Data preparation
@timeit
def prepare_data():
    dataset = EmailDataset(csv_file='email_dataset.csv')

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset

(train_loader, test_loader, train_dataset, test_dataset), prep_time = prepare_data()

# Les neuronnnnns - MLP
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # 2 classes: Safe or Phishing
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc3(x)

input_dim = train_dataset[0][0].shape[0]
model = Net(input_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

@timeit
def train_model(model, train_loader, epochs=15):
    train_losses = []
    model.train()

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        epoch_time = time.time() - epoch_start

        print(f"[+] Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Time: {epoch_time:.2f}s")

    return train_losses

print("[*] Training initial model...")
train_losses, train_time = train_model(model, train_loader)

# visu
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['Training Time'], [train_time], color='blue')
plt.ylabel('Time (seconds)')
plt.title('Training Execution Time')
plt.tight_layout()
plt.savefig('img/training_metrics.png')

# fonction d'évaluation
@timeit
def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    total_time = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            batch_time = time.time() - start_time
            total_time += batch_time

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    cm = confusion_matrix(all_labels, all_preds)
    avg_inference_time = total_time / len(dataloader)

    print("\n[*] Clean evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    return accuracy, cm, avg_inference_time

(clean_acc, clean_cm, clean_time), eval_time = evaluate_model(model, test_loader, device)

# les matriceuuuuu
plt.figure(figsize=(8, 6))
sns.heatmap(clean_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Safe', 'Phishing'],
            yticklabels=['Safe', 'Phishing'])
plt.title('Initial Confusion Matrix\nAccuracy: {:.2f}%'.format(clean_acc * 100))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('img/initial_confusion_matrix.png')

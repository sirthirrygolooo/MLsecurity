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
from copy import deepcopy

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

# Verification GPU
if torch.cuda.is_available():
    print(f"[*] GPU Name: {torch.cuda.get_device_name(device)}")
    print(f"[*] CUDA Version: {torch.version.cuda}")
    print(f"[*] GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.2f} GB")

# Classe pour adapter au dataset
class EmailDataset(Dataset):
    def __init__(self, csv_file, poisoned=False, poison_ratio=0.1, target_class=1):
        self.data = pd.read_csv(csv_file)
        self.poisoned = poisoned
        self.poison_ratio = poison_ratio
        self.target_class = target_class

        if poisoned:
            self.poison_data()

    def poison_data(self):
        num_samples = len(self.data)
        num_poison = int(self.poison_ratio * num_samples)
        poison_indices = np.random.choice(num_samples, num_poison, replace=False)

        for idx in poison_indices:
            features = self.data.iloc[idx, :-1].values.astype(np.float32)
            features[0] *= 1.5
            features[1] += 3
            features[2] = min(features[2] * 2, 1)
            noise = np.random.normal(0, 0.1, len(features))
            features += noise
            self.data.iloc[idx, :-1] = features.astype(np.float32)
            self.data.iloc[idx, -1] = self.target_class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = self.data.iloc[index, :-1].values.astype(np.float32)
        label = self.data.iloc[index, -1]
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)

# Data preparation
@timeit
def prepare_data(poisoned=False, poison_ratio=0.1):
    clean_dataset = EmailDataset(csv_file='email-phishing-dataset/email_phishing_data.csv')
    poisoned_dataset = EmailDataset(csv_file='email-phishing-dataset/email_phishing_data.csv',
                                    poisoned=True, poison_ratio=poison_ratio)

    # Split datasets
    train_size = int(0.8 * len(clean_dataset))
    test_size = len(clean_dataset) - train_size

    if poisoned:
        # Utiliser le dataset empoisonné pour l'entraînement
        train_dataset, test_dataset = torch.utils.data.random_split(poisoned_dataset, [train_size, test_size])
    else:
        train_dataset, test_dataset = torch.utils.data.random_split(clean_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset

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

@timeit
def train_model(model, train_loader, epochs=5):
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

        print(
            f"[+] Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Time: {epoch_time:.2f}s")

    return train_losses

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

    print("\n[*] Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    return accuracy, cm, avg_inference_time

def plot_metrics(train_losses, cm, accuracy, title_suffix=""):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Over Epochs {title_suffix}')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Safe', 'Phishing'],
                yticklabels=['Safe', 'Phishing'])
    plt.title(f'Confusion Matrix {title_suffix}\nAccuracy: {accuracy:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.tight_layout()
    plt.savefig(f'./results/poisoning/metrics_{title_suffix.lower().replace(" ", "_")}.png')

def defense_data_cleaning(dataset, model, device, threshold=0.5):
    clean_data = []
    suspicious_indices = []

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)

            max_probs, _ = torch.max(probabilities, dim=1)
            suspicious = (max_probs < threshold).cpu().numpy()

            for j in range(inputs.size(0)):
                if not suspicious[j]:
                    clean_data.append((inputs[j].cpu().numpy().astype(np.float32), labels[j].item()))
                else:
                    suspicious_indices.append(i * 32 + j)

    clean_features = np.array([x[0] for x in clean_data])
    clean_labels = np.array([x[1] for x in clean_data])

    clean_df = pd.DataFrame(clean_features)
    clean_df['label'] = clean_labels

    return EmailDataset.from_df(clean_df), suspicious_indices

# Ajout d'une méthode from_df pour notre classe Dataset
@classmethod
def from_df(cls, df):
    instance = cls.__new__(cls)
    instance.data = df
    instance.poisoned = False
    instance.poison_ratio = 0
    instance.target_class = None
    return instance

EmailDataset.from_df = from_df

## Expérience 1: Modèle propre
print("\n" + "=" * 50)
print("[*] Entraînement du modèle initial (non empoisonné)")
print("=" * 50 + "\n")

# Préparation des données propres
(train_loader, test_loader, train_dataset, test_dataset), prep_time = prepare_data(poisoned=False)

# Création et entraînement du modèle
input_dim = train_dataset[0][0].shape[0]
clean_model = Net(input_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(clean_model.parameters(), lr=0.001)

print("[*] Training clean model...")
clean_train_losses, clean_train_time = train_model(clean_model, train_loader)

# Évaluation
print("\n[*] Evaluating clean model on clean test data...")
(clean_acc, clean_cm, clean_time), _ = evaluate_model(clean_model, test_loader, device)
plot_metrics(clean_train_losses, clean_cm, clean_acc * 100, "Clean Model")

## Expérience 2: Modèle empoisonné
print("\n" + "=" * 50)
print("[*] Entraînement du modèle empoisonné (10% de données empoisonnées)")
print("=" * 50 + "\n")

# Préparation des données empoisonnées
poison_ratio = 0.1  # 10% des données d'entraînement sont empoisonnées
(poisoned_train_loader, poisoned_test_loader, poisoned_train_dataset, poisoned_test_dataset), _ = prepare_data(
    poisoned=True, poison_ratio=poison_ratio)

# Création et entraînement du modèle empoisonné
poisoned_model = Net(input_dim).to(device)
optimizer = optim.Adam(poisoned_model.parameters(), lr=0.001)

print("[*] Training poisoned model...")
poisoned_train_losses, poisoned_train_time= train_model(poisoned_model, poisoned_train_loader)

# Évaluation sur des données propres
print("\n[*] Evaluating poisoned model on clean test data...")
# Nous utilisons le test_loader original (non empoisonné) pour l'évaluation
(poisoned_acc, poisoned_cm, poisoned_time), _ = evaluate_model(poisoned_model, test_loader, device)
plot_metrics(poisoned_train_losses, poisoned_cm, poisoned_acc * 100, "Poisoned Model")

## Expérience 3: Défense par nettoyage des données
print("\n" + "=" * 50)
print("[*] Test de la défense par nettoyage des données")
print("=" * 50 + "\n")

print("[*] Applying data cleaning defense...")
cleaned_dataset, suspicious_indices = defense_data_cleaning(
    poisoned_train_dataset, clean_model, device, threshold=0.7)

print(
    f"[*] Found {len(suspicious_indices)} suspicious samples ({(len(suspicious_indices) / len(poisoned_train_dataset)) * 100:.2f}%)")

cleaned_train_loader = DataLoader(cleaned_dataset, batch_size=32, shuffle=True)

defended_model = Net(input_dim).to(device)
optimizer = optim.Adam(defended_model.parameters(), lr=0.001)

print("[*] Training defended model...")
(defended_train_losses, defended_train_time), _ = train_model(defended_model, cleaned_train_loader)

print("\n[*] Evaluating defended model on clean test data...")
(defended_acc, defended_cm, defended_time), _ = evaluate_model(defended_model, test_loader, device)
plot_metrics(defended_train_losses, defended_cm, defended_acc * 100, "Defended Model")

print("\n" + "=" * 50)
print("[*] Analyse comparative des modèles")
print("=" * 50 + "\n")

results = pd.DataFrame({
    'Model': ['Clean', 'Poisoned', 'Defended'],
    'Accuracy': [clean_acc, poisoned_acc, defended_acc],
    'Training Time': [clean_train_time, poisoned_train_time, defended_train_time],
    'Inference Time': [clean_time, poisoned_time, defended_time]
})

print(results)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(results['Model'], results['Accuracy'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.bar(results['Model'], results['Training Time'])
plt.title('Training Time Comparison')
plt.ylabel('Time (seconds)')

plt.tight_layout()
plt.savefig('./results/poisoning/model_comparison.png')

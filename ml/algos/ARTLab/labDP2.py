import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import save_image
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {device}")
if torch.cuda.is_available():
    print(f"[*] GPU Name: {torch.cuda.get_device_name(device)}")
    print(f"[*] CUDA Version: {torch.version.cuda}")
    print(f"[*] GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.2f} GB")

# Dataset pour Brain MRI
class BrainMRIDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        folder_name = img_name.split('-')[0]
        img_path = os.path.join(self.root_dir, folder_name, img_name + '.png')
        image = Image.open(img_path).convert('L')
        y_label = torch.tensor(self.annotations.iloc[index, 1])

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

# Transformations améliorées pour augmenter les données
transform = transforms.Compose([
    transforms.Resize((160, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Modèle CNN amélioré avec Batch Normalization
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 40 * 64, 512)
        self.fc2 = nn.Linear(512, 5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 40 * 64)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

# Entraînement avec défense par formation robuste et adversarial training
def robust_train_model(model, train_loader, test_loader, epochs=15, lr=0.001, alpha=0.5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = correct / total
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Test Acc: {test_acc:.4f}")

    return train_losses, test_accuracies

# Méthode pour afficher les courbes ROC
def plot_roc_curve(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_labels, all_preds, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Méthode pour détecter les attaques via DBSCAN
def detect_poisoned_data(model, train_loader, eps=0.3, min_samples=10):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, label in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(label.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(features)

    # Détecter les anomalies
    outliers = np.where(clusters == -1)[0]
    return outliers

# Fonction principale pour la mise en situation
def main():
    print("\n=== Preparing Data ===")
    dataset = BrainMRIDataset(csv_file='adni_dataset/train.csv',
                              root_dir='adni_dataset/ADNI_IMAGES/png_images',
                              transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # validation set
    val_size = int(0.5 * test_size)
    test_size = test_size - val_size
    test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [test_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("\n=== Training Clean Model ===")
    clean_model = Net().to(device)
    clean_losses, clean_accs = robust_train_model(clean_model, train_loader, test_loader)

    clean_acc, clean_cm, clean_report = evaluate_model(clean_model, test_loader)
    print(f"\nClean Model Accuracy: {clean_acc:.4f}")
    print(clean_report)

    # Ajout d'une détection d'attaques par DBSCAN
    print("\n=== Detecting Poisoned Data ===")
    poisoned_data_indices = detect_poisoned_data(clean_model, train_loader)
    print(f"Detected {len(poisoned_data_indices)} poisoned samples using DBSCAN.")

    plot_roc_curve(clean_model, test_loader)

if __name__ == "__main__":
    main()

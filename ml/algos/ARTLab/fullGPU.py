import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import time

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer

from torch.amp import GradScaler, autocast

# Fix seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Utilisation de l'appareil : {device}")
if torch.cuda.is_available():
    print(f"[*] Nom du GPU : {torch.cuda.get_device_name(device)}")

# === Dataset personnalisé ===
class BrainMRIDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Vérification des colonnes dans le CSV
        print("Colonnes dans le CSV:", self.annotations.columns)

        # Vérification des premières lignes pour voir la structure des données
        print("Premières lignes du DataFrame:\n", self.annotations.head())

        # Mapping des diagnostics en labels numériques
        self.annotations['label'] = self.annotations['diagnosis'].map(
            {0: 0, 1: 1, 2: 1, 3: 1, 4: 1})  # Adapté à vos valeurs

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]  # id_code
        folder_name = img_name.split('-')[0]
        img_path = os.path.join(self.root_dir, folder_name, img_name + '.png')
        image = Image.open(img_path).convert('L')

        # Utiliser l'index de la colonne 'label' directement
        y_label = torch.tensor(self.annotations.iloc[index, -1], dtype=torch.long)  # Dernière colonne 'label'

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


# Prétraitement
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Chargement des données
dataset = BrainMRIDataset(csv_file='adni_dataset/train.csv',
                          root_dir='adni_dataset/ADNI_IMAGES/png_images',
                          transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# === Modèle CNN ===
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 classes: Sain (0) et Malade (1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()

# === Entraînement initial avec AMP ===
print("[*] Entraînement du modèle initial (avec AMP)...")
train_losses = []
train_times = []
for epoch in range(10):
    model.train()
    start_time = time.time()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())
    end_time = time.time()
    train_times.append(end_time - start_time)
    print(f"[+] Epoch {epoch + 1}/10 - Loss: {loss.item():.4f} - Time: {end_time - start_time:.2f}s")
    torch.cuda.empty_cache()

# Création dossier de sortie
os.makedirs("img", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss Over Batches')
plt.legend()
plt.savefig('img/training_loss.png')

# === Évaluation initiale ===
model.eval()
with torch.no_grad():
    test_inputs, test_labels = next(iter(test_loader))
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
    predictions = model(test_inputs)
    clean_acc = (torch.argmax(predictions, 1) == test_labels).float().mean()
    print(f"\n[*] Accuracy initiale: {clean_acc:.4f}")

# Matrice de confusion initiale
conf_matrix = confusion_matrix(test_labels.cpu(), torch.argmax(predictions, 1).cpu())
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.annotations['label'].unique(), yticklabels=dataset.annotations['label'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Initial Model')
plt.savefig('img/confusion_matrix_initial.png')

# === Intégration avec ART ===
art_classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 128, 128),
    nb_classes=2,  # 2 classes: Sain (0) et Malade (1)
    clip_values=(0, 1)
)

# === Attaques par évasion ===
def test_evasion_attack(attack, name):
    x_adv = attack.generate(test_inputs.cpu().numpy())
    with torch.no_grad():
        adv_inputs = torch.FloatTensor(x_adv).to(device)
        predictions = model(adv_inputs)
        acc = (torch.argmax(predictions, 1) == test_labels).float().mean()
    print(f"[*] Accuracy après {name}: {acc:.4f}")
    return x_adv

print("\n=== Attaques par évasion ===")
fgsm = FastGradientMethod(art_classifier, eps=0.2)
pgd = ProjectedGradientDescent(art_classifier, eps=0.2, max_iter=10)

x_adv_fgsm = test_evasion_attack(fgsm, "FGSM")
x_adv_pgd = test_evasion_attack(pgd, "PGD")

# === Visualisation des attaques ===
def plot_attack_samples(original, adversarial, title, filename):
    plt.figure(figsize=(10, 4))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(original[i].cpu().squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(adversarial[i].squeeze(), cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'img/{filename}')

plot_attack_samples(test_inputs, x_adv_fgsm, "FGSM", "fgsm_attack.png")
plot_attack_samples(test_inputs, x_adv_pgd, "PGD", "pgd_attack.png")

# === Défense : entraînement adversarial ===
print("\n=== Défenses ===")
trainer = AdversarialTrainer(art_classifier, attacks=[fgsm, pgd], ratio=0.5)
trainer.fit(train_dataset.dataset.annotations.iloc[:, 0].values, train_dataset.dataset.annotations.iloc[:, 1].values, nb_epochs=15)

# === Évaluation finale ===
model.eval()
with torch.no_grad():
    test_inputs, test_labels = next(iter(test_loader))
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
    predictions = model(test_inputs)
    final_acc = (torch.argmax(predictions, 1) == test_labels).float().mean()
    print(f"\n[*] Accuracy finale après défenses: {final_acc:.4f}")

# Matrice de confusion finale
conf_matrix_final = confusion_matrix(test_labels.cpu(), torch.argmax(predictions, 1).cpu())
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_final, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.annotations['label'].unique(), yticklabels=dataset.annotations['label'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Final Model')
plt.savefig('img/confusion_matrix_final.png')

# Analyse de corrélation
correlation_matrix = pd.DataFrame(train_dataset.dataset.annotations).corr()
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('img/correlation_matrix.png')

# Analyse de temps d'exécution
plt.figure(figsize=(10, 5))
plt.plot(train_times, label='Training Time per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Time (s)')
plt.title('Training Time per Epoch')
plt.legend()
plt.savefig('img/training_time.png')

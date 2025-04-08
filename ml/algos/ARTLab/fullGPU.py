import os
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    from art.attacks.poisoning import PoisoningAttackBackdoor
    from art.attacks.poisoning.perturbations import add_pattern_bd
    from art.defences.trainer import AdversarialTrainer
except ImportError:
    print('Import error')
    exit(1)

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

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        folder_name = img_name.split('-')[0]
        img_path = os.path.join(self.root_dir, folder_name, img_name + '.png')
        image = Image.open(img_path).convert('L')  # grayscale
        y_label = torch.tensor(self.annotations.iloc[index, 1])

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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# === Modèle CNN ===
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Entraînement initial ===
print("[*] Entraînement du modèle initial...")
train_losses = []
for epoch in range(10):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
    print(f"[+] Epoch {epoch + 1}/10 - Loss: {loss.item():.4f}")

# Création dossier de sortie
os.makedirs("img", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss Over Batches')
plt.legend()
plt.savefig('img/_full.png')

# === Évaluation initiale ===
model.eval()
with torch.no_grad():
    test_inputs, test_labels = next(iter(test_loader))
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
    predictions = model(test_inputs)
    clean_acc = (torch.argmax(predictions, 1) == test_labels).float().mean()
    print(f"\n[*] Accuracy initiale: {clean_acc:.4f}")

# === Intégration avec ART ===
art_classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 128, 128),
    nb_classes=5,
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

plot_attack_samples(test_inputs, x_adv_fgsm, "FGSM", "fgsm_attack_full.png")
plot_attack_samples(test_inputs, x_adv_pgd, "PGD", "pgd_attack_full.png")

# === Attaque par empoisonnement ===
print("\n=== Attaque par empoisonnement ===")
backdoor_attack = PoisoningAttackBackdoor(perturbation=add_pattern_bd)
n_poison = int(len(train_dataset) * 0.1)
poison_indices = np.random.choice(len(train_dataset), n_poison, replace=False)

# Convertir les images en niveaux de gris en images en couleur
x_poison_color = [np.stack((image,)*3, axis=-1) for image in [train_dataset[i][0].numpy() for i in poison_indices]]
y_poison = np.full(n_poison, 0)

x_poison_adv, y_poison_adv = backdoor_attack.poison(np.array(x_poison_color), y_poison)

# === Réentraînement avec backdoor ===
print("[*] Entraînement avec backdoor...")
X_train_clean = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
y_train_clean = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

# Convertir les images empoisonnées en niveaux de gris
x_poison_adv_gray = [np.mean(image, axis=-1, keepdims=True) for image in x_poison_adv]
X_train_clean[poison_indices] = torch.tensor(np.array(x_poison_adv_gray)).squeeze(1)
y_train_clean[poison_indices] = torch.tensor(y_poison_adv)

for epoch in range(10):
    model.train()
    inputs = X_train_clean.to(device)
    labels = y_train_clean.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# === Test du backdoor ===
x_test_backdoor, _ = backdoor_attack.poison(test_inputs[:5].cpu().numpy(), np.zeros(5))
with torch.no_grad():
    predictions = model(torch.FloatTensor(x_test_backdoor).to(device))
    print("[*] Prédictions backdoor:", torch.argmax(predictions, 1).cpu().numpy())

plot_attack_samples(test_inputs[:5], x_test_backdoor, "Backdoor", "backdoor_attack_full.png")

# === Défense : entraînement adversarial ===
print("\n=== Défenses ===")
trainer = AdversarialTrainer(art_classifier, attacks=[fgsm, pgd], ratio=0.5)
trainer.fit(X_train_clean.numpy(), y_train_clean.numpy(), nb_epochs=15)

# === Évaluation finale ===
model.eval()
with torch.no_grad():
    test_inputs, test_labels = next(iter(test_loader))
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
    predictions = model(test_inputs)
    final_acc = (torch.argmax(predictions, 1) == test_labels).float().mean()
    print(f"\n[*] Accuracy finale après défenses: {final_acc:.4f}")

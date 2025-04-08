# Demonstration basique d'ART 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.defences.trainer import AdversarialTrainer

# Configuration pour reproductibilité
torch.manual_seed(42)
np.random.seed(42)

# Chargement des données
digits = load_digits()
X = digits.images / 16.0  # Normalisation [0-1]
y = digits.target

# Conversion en tenseurs PyTorch
X_tensor = torch.FloatTensor(X).permute(0, 2, 1)  # Format (N, H, W)
y_tensor = torch.LongTensor(y)

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X_tensor.numpy(), y_tensor.numpy(), test_size=0.2, random_state=42)

# Modèle PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Initialisation du modèle
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement de base
print("Entraînement du modèle initial...")
train_losses = []
for epoch in range(10):
    inputs = torch.FloatTensor(X_train)
    labels = torch.LongTensor(y_train)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch + 1}/10 - Loss: {loss.item():.4f}")

# Sauvegarde du graphique de la perte
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.savefig('img/training_loss.png')

# Évaluation
with torch.no_grad():
    test_inputs = torch.FloatTensor(X_test)
    predictions = model(test_inputs)
    clean_acc = (torch.argmax(predictions, 1) == torch.LongTensor(y_test)).float().mean()
    print(f"\nAccuracy initiale: {clean_acc:.4f}")

# Configuration ART
art_classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(8, 8),
    nb_classes=10,
    clip_values=(0, 1)
)

# Attaques par évasion
def test_evasion_attack(attack, name):
    x_adv = attack.generate(X_test)

    with torch.no_grad():
        adv_inputs = torch.FloatTensor(x_adv)
        predictions = model(adv_inputs)
        acc = (torch.argmax(predictions, 1) == torch.LongTensor(y_test)).float().mean()

    print(f"Accuracy après {name}: {acc:.4f}")
    return x_adv

print("\n=== Attaques par évasion ===")
fgsm = FastGradientMethod(art_classifier, eps=0.2)
pgd = ProjectedGradientDescent(art_classifier, eps=0.2, max_iter=10)

x_adv_fgsm = test_evasion_attack(fgsm, "FGSM")
x_adv_pgd = test_evasion_attack(pgd, "PGD")

# Visualisation
def plot_attack_samples(original, adversarial, title, filename):
    plt.figure(figsize=(10, 4))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(original[i], cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(adversarial[i], cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'img/{filename}')

plot_attack_samples(X_test, x_adv_fgsm, "FGSM", "fgsm_attack.png")
plot_attack_samples(X_test, x_adv_pgd, "PGD", "pgd_attack.png")

# Attaque par empoisonnement
print("\n=== Attaque par empoisonnement ===")
backdoor_attack = PoisoningAttackBackdoor(perturbation=add_pattern_bd)
n_poison = int(len(X_train) * 0.1)
poison_indices = np.random.choice(len(X_train), n_poison, replace=False)

x_poison = X_train[poison_indices].copy()
y_poison = np.full(n_poison, 0)  # Cible: classe 0

x_poison_adv, y_poison_adv = backdoor_attack.poison(x_poison, y_poison)

# Entraînement avec données empoisonnées
print("Entraînement avec backdoor...")
X_train_poisoned = np.copy(X_train)
X_train_poisoned[poison_indices] = x_poison_adv
y_train_poisoned = np.copy(y_train)
y_train_poisoned[poison_indices] = y_poison_adv

for epoch in range(10):
    inputs = torch.FloatTensor(X_train_poisoned)
    labels = torch.LongTensor(y_train_poisoned)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Test du backdoor
x_test_backdoor, _ = backdoor_attack.poison(X_test[:5], np.zeros(5))
with torch.no_grad():
    predictions = model(torch.FloatTensor(x_test_backdoor))
    print("Prédictions backdoor:", torch.argmax(predictions, 1).numpy())

plot_attack_samples(X_test[:5], x_test_backdoor, "Backdoor", "backdoor_attack.png")

# Défenses
print("\n=== Défenses ===")

# Entraînement adversarial
trainer = AdversarialTrainer(art_classifier, [fgsm, pgd], ratio=0.5)
trainer.fit(np.concatenate([X_train, X_train_poisoned]),
            np.concatenate([y_train, y_train_poisoned]),
            nb_epochs=15)

# Évaluation finale
with torch.no_grad():
    test_inputs = torch.FloatTensor(X_test)
    predictions = model(test_inputs)
    final_acc = (torch.argmax(predictions, 1) == torch.LongTensor(y_test)).float().mean()
    print(f"\nAccuracy finale après défenses: {final_acc:.4f}")

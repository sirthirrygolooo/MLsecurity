import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

project_root = '.'
results_dir = os.path.join(project_root, 'results', 'minilab')
os.makedirs(results_dir, exist_ok=True)

def add_trigger(img, size=3):
    img = img.clone()
    img[0, -size:, -size:] = 1.0  # lo poti carré
    return img

class PoisonedMNIST(Dataset):
    def __init__(self, poison_rate=0.1, target_label=7):
        transform = transforms.ToTensor()
        mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.data = []
        self.targets = []

        for i in range(len(mnist)):
            img, label = mnist[i]

            if random.random() < poison_rate:
                # Ajouter trigger et changer la classe
                img = add_trigger(img)
                label = target_label
            self.data.append(img)
            self.targets.append(label)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 100), nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.net(x)

def train(model, dataloader, epochs=3):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    losses = []
    accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            _, predicted = torch.max(out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        losses.append(epoch_loss / len(dataloader))
        accuracies.append(correct / total)
        print(f"Epoch {epoch + 1} done, Loss: {epoch_loss / len(dataloader):.4f}, Accuracy: {correct / total:.2%}")

    return losses, accuracies

def test_backdoor(model):
    model.eval()
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    correct_clean = 0
    correct_poisoned = 0
    total = 0
    target_label = 7
    all_labels = []
    all_preds_clean = []
    all_preds_poisoned = []

    for i in range(1000):
        img, label = mnist[i]
        total += 1
        img_clean = img.unsqueeze(0).to(device)
        img_poisoned = add_trigger(img).unsqueeze(0).to(device)

        pred_clean = model(img_clean).argmax(1).item()
        pred_poisoned = model(img_poisoned).argmax(1).item()

        all_labels.append(label)
        all_preds_clean.append(pred_clean)
        all_preds_poisoned.append(pred_poisoned)

        if pred_clean == label:
            correct_clean += 1
        if pred_poisoned == target_label:
            correct_poisoned += 1

    print(f"Accuracy sur données propres : {correct_clean / total:.2%}")
    print(f"Attaque backdoor (trigger → {target_label}) : succès à {correct_poisoned / total:.2%}")

    return all_labels, all_preds_clean, all_preds_poisoned

def plot_examples(dataset, n=5, save_path=os.path.join(results_dir, 'examples.png')):
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for i in range(n):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.savefig(save_path)
    plt.close()

def plot_loss_accuracy(losses, accuracies, save_path=os.path.join(results_dir, 'loss_accuracy.png')):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(losses, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(accuracies, color=color, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(labels, preds, title, save_path):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def reverse_engineer_trigger(model, dataloader, epochs=5, lr=0.1):
    model.eval()
    trigger_pattern = torch.randn((1, 28, 28), requires_grad=True, device=device)
    optimizer = optim.Adam([trigger_pattern], lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            trigger_pattern.data.clamp_(0, 1)  # Ensure the trigger pattern is within valid image range

            # Apply the trigger pattern to the images
            imgs_with_trigger = imgs + trigger_pattern
            imgs_with_trigger = torch.clamp(imgs_with_trigger, 0, 1)

            # Forward pass
            outputs = model(imgs_with_trigger)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    return trigger_pattern

def mitigate_trigger(model, dataloader, trigger_pattern, epochs=5, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Apply the trigger pattern to the images
            imgs_with_trigger = imgs + trigger_pattern
            imgs_with_trigger = torch.clamp(imgs_with_trigger, 0, 1)

            # Forward pass
            outputs = model(imgs_with_trigger)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
poisoned_dataset = PoisonedMNIST(poison_rate=0.1)
train_loader = DataLoader(poisoned_dataset, batch_size=64, shuffle=True)

print("Exemples d'images avec déclencheur :")
plot_examples(poisoned_dataset)

losses, accuracies = train(model, train_loader)

plot_loss_accuracy(losses, accuracies)

labels, preds_clean, preds_poisoned = test_backdoor(model)

plot_confusion_matrix(labels, preds_clean, "Matrice de confusion (données propres)", os.path.join(results_dir, 'confusion_matrix_clean.png'))
plot_confusion_matrix(labels, preds_poisoned, "Matrice de confusion (données empoisonnées)", os.path.join(results_dir, 'confusion_matrix_poisoned.png'))

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    img, label = poisoned_dataset[i]
    img_poisoned = add_trigger(img)

    pred_clean = model(img.unsqueeze(0).to(device)).argmax(1).item()
    pred_poisoned = model(img_poisoned.unsqueeze(0).to(device)).argmax(1).item()

    axes[0, i].imshow(img.squeeze(), cmap='gray')
    axes[0, i].set_title(f'Clean: {pred_clean}')
    axes[0, i].axis('off')

    axes[1, i].imshow(img_poisoned.squeeze(), cmap='gray')
    axes[1, i].set_title(f'Poisoned: {pred_poisoned}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'prediction_comparison.png'))
plt.close()

# Reverse engineer the trigger pattern
trigger_pattern = reverse_engineer_trigger(model, train_loader)

# Mitigate the trigger pattern
mitigate_trigger(model, train_loader, trigger_pattern)

# Re-evaluate the model after mitigation
labels, preds_clean, preds_poisoned = test_backdoor(model)

plot_confusion_matrix(labels, preds_clean, "Matrice de confusion (données propres après mitigation)", os.path.join(results_dir, 'confusion_matrix_clean_mitigated.png'))
plot_confusion_matrix(labels, preds_poisoned, "Matrice de confusion (données empoisonnées après mitigation)", os.path.join(results_dir, 'confusion_matrix_poisoned_mitigated.png'))

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

# Configuration
config = {
    "project_root": ".",
    "results_dir": "./results/minilab",
    "batch_size": 64,
    "poison_rate": 0.1,
    "target_label": 7,
    "epochs": 3,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

if torch.cuda.is_available():
    print(f"[*] Using GPU: {torch.cuda.get_device_name(config['device'])}")
else:
    print("[*] Using CPU")

os.makedirs(config['results_dir'], exist_ok=True)

# Utils

def add_trigger(img, size=3):
    img = img.clone()
    img[0, -size:, -size:] = 1.0
    return img

class PoisonedMNIST(Dataset):
    def __init__(self, poison_rate=0.1, target_label=7, train=True, transform=None):
        if transform is None:
            transform = transforms.ToTensor()
        mnist = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        self.data = []
        self.targets = []

        for i in range(len(mnist)):
            img, label = mnist[i]
            if random.random() < poison_rate:
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
    losses, accuracies = [], []

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(config['device']), labels.to(config['device'])
            out = model(imgs)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)

        losses.append(total_loss / len(dataloader))
        accuracies.append(correct / total)
        print(f"Epoch {epoch + 1}, Loss: {losses[-1]:.4f}, Accuracy: {accuracies[-1]:.2%}")

    return losses, accuracies

# Evaluation

def test_backdoor(model):
    model.eval()
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    correct_clean, correct_poisoned, total = 0, 0, 0
    labels, preds_clean, preds_poisoned = [], [], []

    for i in range(1000):
        img, label = mnist[i]
        img_clean = img.unsqueeze(0).to(config['device'])
        img_poisoned = add_trigger(img).unsqueeze(0).to(config['device'])

        pred_clean = model(img_clean).argmax(1).item()
        pred_poisoned = model(img_poisoned).argmax(1).item()

        labels.append(label)
        preds_clean.append(pred_clean)
        preds_poisoned.append(pred_poisoned)

        correct_clean += pred_clean == label
        correct_poisoned += pred_poisoned == config['target_label']
        total += 1

    print(f"Accuracy (clean): {correct_clean / total:.2%}")
    print(f"Backdoor success rate: {correct_poisoned / total:.2%}")
    print("F1 Score (clean):", f1_score(labels, preds_clean, average='macro'))

    return labels, preds_clean, preds_poisoned

def plot_examples(dataset, n=5, save_path=None):
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for i in range(n):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_loss_accuracy(losses, accuracies, save_path):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='red')
    ax1.plot(losses, color='red')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='blue')
    ax2.plot(accuracies, color='blue')
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

# Trigger Reverse Engineering

def reverse_engineer_trigger(model, dataloader, epochs=5, lr=0.1):
    model.eval()
    trigger_pattern = torch.randn((1, 28, 28), requires_grad=True, device=config['device'])
    optimizer = optim.Adam([trigger_pattern], lr=lr)
    mask = torch.zeros_like(trigger_pattern)
    mask[:, -3:, -3:] = 1.0

    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(config['device']), labels.to(config['device'])
            trigger_pattern.data.clamp_(0, 1)
            imgs_with_trigger = torch.clamp(imgs + trigger_pattern * mask, 0, 1)
            outputs = model(imgs_with_trigger)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Reverse Engineering Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    return trigger_pattern * mask

# Mitigation
def mitigate_trigger(model, dataloader, trigger_pattern, epochs=5, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(config['device']), labels.to(config['device'])
            imgs_with_trigger = torch.clamp(imgs + trigger_pattern.detach(), 0, 1)
            outputs = model(imgs_with_trigger)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Mitigation Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# Main Pipeline

model = SimpleCNN().to(config['device'])
poisoned_dataset = PoisonedMNIST(poison_rate=config['poison_rate'], target_label=config['target_label'])
train_loader = DataLoader(poisoned_dataset, batch_size=config['batch_size'], shuffle=True)

plot_examples(poisoned_dataset, save_path=os.path.join(config['results_dir'], 'examples.png'))
losses, accuracies = train(model, train_loader, epochs=config['epochs'])
plot_loss_accuracy(losses, accuracies, os.path.join(config['results_dir'], 'loss_accuracy.png'))

labels, preds_clean, preds_poisoned = test_backdoor(model)
plot_confusion_matrix(labels, preds_clean, "Confusion matrix (clean)", os.path.join(config['results_dir'], 'confusion_matrix_clean.png'))
plot_confusion_matrix(labels, preds_poisoned, "Confusion matrix (poisoned)", os.path.join(config['results_dir'], 'confusion_matrix_poisoned.png'))

trigger_pattern = reverse_engineer_trigger(model, train_loader)
mitigate_trigger(model, train_loader, trigger_pattern)

# Re-evaluate
labels, preds_clean, preds_poisoned = test_backdoor(model)
plot_confusion_matrix(labels, preds_clean, "Clean after mitigation", os.path.join(config['results_dir'], 'confusion_matrix_clean_mitigated.png'))
plot_confusion_matrix(labels, preds_poisoned, "Poisoned after mitigation", os.path.join(config['results_dir'], 'confusion_matrix_poisoned_mitigated.png'))
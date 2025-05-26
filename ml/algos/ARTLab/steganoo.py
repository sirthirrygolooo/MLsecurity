import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, auc
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = 'results/steg/test/'
os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

steg_dataset_path = 'steg_dataset'
excel_path = os.path.join(steg_dataset_path, 'train.csv')
img_path = os.path.join(steg_dataset_path, 'img')

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

annotations = pd.read_csv(excel_path)
unique_labels = annotations['diagnosis'].unique()

label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def estimate_noise(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])
        high_pass = cv2.filter2D(image, -1, kernel)

        noise_level = np.var(high_pass)
        return noise_level

    def add_noise(self, image, noise_level):
        if isinstance(image, Image.Image):
            image = np.array(image)

        noise_amount = 10 / (noise_level + 1e-6)

        noise = np.random.normal(0, noise_amount, image.shape)

        noisy_image = image + noise

        noisy_image = np.clip(noisy_image, 0, 255)

        noisy_image = Image.fromarray(noisy_image.astype('uint8'))

        return noisy_image

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f"image_{index}.png")
        image = Image.open(img_path).convert('L')
        label = self.annotations.iloc[index]['diagnosis']

        noise_level = self.estimate_noise(image)

        # Add noise proportionally
        noisy_image = self.add_noise(image, noise_level)

        if self.transform:
            noisy_image = self.transform(noisy_image)

        label_index = label_to_index[label]
        return noisy_image, torch.tensor(label_index, dtype=torch.long)

dataset = CustomDataset(csv_file=excel_path, img_dir=img_path, transform=transform)

train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.3, random_state=42)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class Net(nn.Module):
    """CNN model for classification."""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 128, 128)
            dummy_output = self.pool(torch.relu(self.conv2(self.pool(torch.relu(self.conv1(dummy_input))))))
            flattened_size = dummy_output.view(-1).shape[0]

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, len(unique_labels))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

def train_model(model, train_loader, criterion, optimizer, epochs=15):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
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
        train_accuracies.append(epoch_acc)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    test_losses.append(epoch_loss)
    test_accuracies.append(epoch_acc)
    print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.4f}")

    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"F1 Score: {f1:.4f}")

    print(classification_report(all_labels, all_predictions, target_names=unique_labels))

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('%sconfusion_matrix.png' % RESULTS_DIR)
    plt.close()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(unique_labels)):
        fpr[i], tpr[i], _ = roc_curve(np.array(all_labels) == i, np.array(all_probabilities)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'cyan']
    for i, color in zip(range(len(unique_labels)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(unique_labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.savefig('%sroc_curves.png' % RESULTS_DIR)
    plt.close()

train_model(model, train_loader, criterion, optimizer, epochs=15)
evaluate_model(model, test_loader, criterion)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('%straining_validation_loss_accuracy.png' % RESULTS_DIR)

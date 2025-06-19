import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, auc
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

OUTPUT_DIR = 'results/steg/comparison/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets = {
    'clean': 'steg_dataset_clean',
    'attacked': 'steg_dataset',
    'stegano': 'noise_stegano',
}

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 128, 128)
            dummy_output = self.pool(torch.relu(self.conv2(self.pool(torch.relu(self.conv1(dummy_input))))))
            flattened_size = dummy_output.view(-1).shape[0]

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f"image_{index}.png")
        image = Image.open(img_path).convert('L')
        label = self.annotations.iloc[index]['diagnosis']
        if self.transform:
            image = self.transform(image)
        label_index = label_to_index[label]
        return image, torch.tensor(label_index, dtype=torch.long)

def train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    train_losses = []
    train_accuracies = []

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

    return train_losses, train_accuracies

def evaluate_model(model, test_loader, criterion, dataset_name):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
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

    f1 = f1_score(all_labels, all_predictions, average='weighted')

    metrics = {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'f1_score': f1,
        'classification_report': classification_report(all_labels, all_predictions, target_names=unique_labels, output_dict=True),
        'confusion_matrix': confusion_matrix(all_labels, all_predictions)
    }

    return metrics

def compare_datasets():
    results = {}

    for dataset_name, dataset_path in datasets.items():
        excel_path = os.path.join(dataset_path, 'train.csv')
        img_path = os.path.join(dataset_path, 'img')

        annotations = pd.read_csv(excel_path)
        global unique_labels
        unique_labels = annotations['diagnosis'].unique()
        global label_to_index
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        dataset = CustomDataset(csv_file=excel_path, img_dir=img_path, transform=transform)

        train_indices, test_indices = train_test_split(
            range(len(dataset)), test_size=0.3, random_state=42
        )
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = Net(num_classes=len(unique_labels)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_losses, train_accuracies = train_model(
            model, train_loader, criterion, optimizer
        )

        metrics = evaluate_model(model, test_loader, criterion, dataset_name)

        results[dataset_name] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'metrics': metrics
        }

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_{dataset_name}.pth'))

    return results

def plot_training_curves(results):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for dataset_name, data in results.items():
        plt.plot(data['train_losses'], label=f'{dataset_name} Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()

    plt.subplot(1, 2, 2)
    for dataset_name, data in results.items():
        plt.plot(data['train_accuracies'], label=f'{dataset_name} Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves_comparison.png'))
    plt.close()

def plot_metrics_comparison(results):
    metrics_names = ['loss', 'accuracy', 'f1_score']
    metrics_values = {
        name: [results[dataset]['metrics'][name] for dataset in datasets]
        for name in metrics_names
    }

    x = np.arange(len(datasets))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6))
    for metric_name, values in metrics_values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=metric_name)
        ax.bar_label(rects, padding=3, fmt='%.4f')
        multiplier += 1

    ax.set_ylabel('Scores')
    ax.set_title('Metrics comparison by dataset')
    ax.set_xticks(x + width, datasets.keys())
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1.1)

    plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_comparison.png'))
    plt.close()

def plot_confusion_matrices(results):
    for dataset_name, data in results.items():
        cm = data['metrics']['confusion_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels, yticklabels=unique_labels
        )
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{dataset_name}.png'))
        plt.close()

def plot_roc_curves(results):
    for dataset_name, data in results.items():
        model = Net(num_classes=len(unique_labels)).to(device)
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'model_{dataset_name}.pth')))
        model.eval()

        test_loader = DataLoader(
            Subset(
                CustomDataset(
                    csv_file=os.path.join(datasets[dataset_name], 'train.csv'),
                    img_dir=os.path.join(datasets[dataset_name], 'img'),
                    transform=transform
                ),
                range(int(0.7 * len(CustomDataset(
                    csv_file=os.path.join(datasets[dataset_name], 'train.csv'),
                    img_dir=os.path.join(datasets[dataset_name], 'img'),
                    transform=transform
                ))), len(CustomDataset(
                    csv_file=os.path.join(datasets[dataset_name], 'train.csv'),
                    img_dir=os.path.join(datasets[dataset_name], 'img'),
                    transform=transform
                )))
            ),
            batch_size=BATCH_SIZE, shuffle=False
        )

        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

        plt.figure(figsize=(8, 6))
        for i in range(len(unique_labels)):
            fpr, tpr, _ = roc_curve(np.array(all_labels) == i, np.array(all_probabilities)[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{unique_labels[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {dataset_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(OUTPUT_DIR, f'roc_curves_{dataset_name}.png'))
        plt.close()

results = compare_datasets()
plot_training_curves(results)
plot_metrics_comparison(results)
plot_confusion_matrices(results)
plot_roc_curves(results)

with open(os.path.join(OUTPUT_DIR, 'comparison_results.txt'), 'w') as f:
    for dataset_name, data in results.items():
        f.write(f"\nDataset: {dataset_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Final Training Accuracy: {data['train_accuracies'][-1]:.4f}\n")
        f.write(f"Test Loss: {data['metrics']['loss']:.4f}\n")
        f.write(f"Test Accuracy: {data['metrics']['accuracy']:.4f}\n")
        f.write(f"F1 Score: {data['metrics']['f1_score']:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(pd.DataFrame(data['metrics']['classification_report']).to_string())
        f.write("\n\n")

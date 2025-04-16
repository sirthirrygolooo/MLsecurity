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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.cluster import KMeans
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

# Toujoursd la même pour le dataset
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


transform = transforms.Compose([
    transforms.Resize((160, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Les neuronnnnns - CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # niveau de gris donc 1 canal entree, 32 canaux sortie, filtre 3*3, padding 1px pour conserver les val
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 32 canaux entree, 64 canaux sortie, filtre 3*3, padding 1px pour conserver les val
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # couche de pooling
        self.pool = nn.MaxPool2d(2, 2)
        # première couche entierement conn.e (64*40*64 entrees, 512 sorties)
        self.fc1 = nn.Linear(64 * 40 * 64, 512)
        # deuxième couche entierement conn. (512 entrees, 5 sorties) -> 5 diagnostics
        self.fc2 = nn.Linear(512, 5)
        # couche de dropout pour eviter l'overfitting (0.5 = 50% des neurones sont mis à 0)
        self.dropout = nn.Dropout(0.5)

    # opération de propagation avant
    def forward(self, x):
        # premiere couche de convolution + activation ReLU + pooling
        x = self.pool(torch.relu(self.conv1(x)))
        # deuxieme couche de coonv.
        x = self.pool(torch.relu(self.conv2(x)))
        # applatir en v. 1d pour appliquer la couche entierement conn.
        x = x.view(-1, 64 * 40 * 64)
        # applique la premiere couche entierement conn. + activation ReLU + dropout
        x = self.dropout(torch.relu(self.fc1(x)))
        # applique la deuxieme couche entierement conn. et return
        return self.fc2(x)


def train_model(model, train_loader, test_loader, epochs=15, lr=0.001):
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

        # evaluation sur test.
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

def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    return accuracy, cm, report


class DataPoisoner:
    def __init__(self, dataset):
        self.dataset = dataset
        self.original_labels = [self.dataset[i][1] for i in range(len(self.dataset))]

    def random_label_flip(self, flip_percentage=0.2, target_class=None):
        num_to_flip = int(len(self.dataset) * flip_percentage)
        indices = np.random.choice(len(self.dataset), num_to_flip, replace=False)

        for idx in indices:
            real_idx = self.dataset.indices[idx]
            original_label = self.dataset.dataset.annotations.iloc[real_idx, 1]

            if target_class is not None:
                new_label = target_class
            else:
                possible_labels = [l for l in range(5) if l != original_label]
                new_label = np.random.choice(possible_labels)

            self.dataset.dataset.annotations.iloc[real_idx, 1] = new_label

        print(f"Flipped {num_to_flip} labels ({flip_percentage * 100}%)")
        return indices

    def backdoor_attack(self, trigger_pattern, target_class, poison_percentage=0.1):
        """Add a backdoor trigger to a subset of images and change their labels"""
        num_to_poison = int(len(self.dataset) * poison_percentage)
        indices = np.random.choice(len(self.dataset), num_to_poison, replace=False)

        for idx in indices:
            image, _ = self.dataset[idx]
            poisoned_image = torch.clamp(image + trigger_pattern, -1, 1)

            real_idx = self.dataset.indices[idx]
            new_img_name = f"poisoned-{idx}"
            folder_name = new_img_name.split('-')[0]

            poisoned_path = os.path.join(
                self.dataset.dataset.root_dir,
                folder_name,
                new_img_name + '.png'
            )

            os.makedirs(os.path.dirname(poisoned_path), exist_ok=True)
            save_image(poisoned_image, poisoned_path)

            self.dataset.dataset.annotations.iloc[real_idx, 0] = new_img_name
            self.dataset.dataset.annotations.iloc[real_idx, 1] = target_class

        print(f"Added backdoor to {num_to_poison} samples ({poison_percentage * 100}%)")
        return indices

    def get_poisoned_dataset(self):
        return self.dataset


class DataPoisoningDefender:
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

    def detect_label_outliers(self, contamination=0.1):
        """Use model's confidence scores to detect potential label outliers"""
        self.model.eval()
        confidences = []
        labels = []

        with torch.no_grad():
            for inputs, label in self.train_loader.dataset:
                inputs = inputs.unsqueeze(0).to(device)
                outputs = self.model(inputs)
                prob = torch.softmax(outputs, dim=1)
                confidence = torch.max(prob).item()
                confidences.append(confidence)
                labels.append(label)

        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        outlier_scores = lof.fit_predict(np.array(confidences).reshape(-1, 1))
        potential_outliers = np.where(outlier_scores == -1)[0]

        return potential_outliers

    def cluster_based_filtering(self, n_clusters=5, threshold=0.9):
        """Cluster samples and filter out small clusters as potential poisoning"""
        self.model.eval()
        features = []
        labels = []

        with torch.no_grad():
            for inputs, label in self.train_loader.dataset:
                inputs = inputs.unsqueeze(0).to(device)
                # Get features from the layer before the final classification
                x = self.model.pool(torch.relu(self.model.conv1(inputs)))
                x = self.model.pool(torch.relu(self.model.conv2(x)))
                features.append(x.cpu().numpy().flatten())
                labels.append(label)

        features = np.array(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)

        cluster_counts = Counter(clusters)
        small_clusters = [c for c, count in cluster_counts.items()
                          if count < threshold * len(features) / n_clusters]

        potential_poison = np.where(np.isin(clusters, small_clusters))[0]
        return potential_poison

    def robust_training(self, epochs=15, lr=0.001, alpha=0.5):
        """Train with a combination of clean and robust losses"""
        criterion_ce = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)

                # Standard cross-entropy loss
                loss_ce = criterion_ce(outputs, labels)

                # Additional robust loss (e.g., emphasizing harder samples)
                probs = torch.softmax(outputs, dim=1)
                loss_robust = -torch.log(1 - probs[torch.arange(len(labels)), labels] + 1e-7).mean()

                # Combined loss
                loss = alpha * loss_ce + (1 - alpha) * loss_robust
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Robust Training Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(self.train_loader):.4f}")


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
    clean_losses, clean_accs = train_model(clean_model, train_loader, test_loader)

    clean_acc, clean_cm, clean_report = evaluate_model(clean_model, test_loader)
    print(f"\nClean Model Accuracy: {clean_acc:.4f}")
    print(clean_report)

    print("\n=== Performing Data Poisoning Attacks ===")
    poisoner = DataPoisoner(train_dataset)

    flipped_indices = poisoner.random_label_flip(flip_percentage=0.2)
    poisoned_train_loader = DataLoader(poisoner.get_poisoned_dataset(), batch_size=32, shuffle=True)

    trigger_pattern = torch.zeros(1, 160, 256)
    trigger_pattern[:, 150:155, 245:250] = 0.5  # Small white square in corner
    backdoor_indices = poisoner.backdoor_attack(trigger_pattern, target_class=0, poison_percentage=0.1)

    print("\n=== Training Model on Poisoned Data ===")
    poisoned_model = Net().to(device)
    poisoned_losses, poisoned_accs = train_model(poisoned_model, poisoned_train_loader, test_loader)

    poisoned_acc, poisoned_cm, poisoned_report = evaluate_model(poisoned_model, test_loader)
    print(f"\nPoisoned Model Accuracy: {poisoned_acc:.4f}")
    print(poisoned_report)

    print("\n=== Evaluating Backdoor Attack ===")
    triggered_test_set = []
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        triggered_img = torch.clamp(img + trigger_pattern, -1, 1)
        triggered_test_set.append((triggered_img, label))
    triggered_loader = DataLoader(triggered_test_set, batch_size=32, shuffle=False)

    poisoned_model.eval()
    target_class = 0
    correct = 0
    total = 0
    target_count = 0

    with torch.no_grad():
        for inputs, labels in triggered_loader:
            inputs = inputs.to(device)
            outputs = poisoned_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            target_count += (predicted.cpu() == target_class).sum().item()

    backdoor_success_rate = target_count / total
    print(f"Backdoor success rate: {backdoor_success_rate:.4f}")

    print("\n=== Implementing Defenses ===")
    defender = DataPoisoningDefender(clean_model, poisoned_train_loader)

    print("\nDetecting label outliers...")
    outlier_indices = defender.detect_label_outliers(contamination=0.15)
    print(f"Detected {len(outlier_indices)} potential outliers")

    true_positives = len(set(outlier_indices) & set(flipped_indices))
    print(f"True positives: {true_positives}/{len(flipped_indices)}")

    print("\nApplying cluster-based filtering...")
    cluster_outliers = defender.cluster_based_filtering()
    print(f"Detected {len(cluster_outliers)} potential poisoning samples via clustering")

    all_outliers = set(outlier_indices).union(set(cluster_outliers))
    clean_indices = [i for i in range(len(poisoned_train_loader.dataset)) if i not in all_outliers]
    sanitized_dataset = Subset(poisoned_train_loader.dataset, clean_indices)
    sanitized_loader = DataLoader(sanitized_dataset, batch_size=32, shuffle=True)

    print("\nTraining robust model...")
    robust_model = Net().to(device)
    defender_robust = DataPoisoningDefender(robust_model, sanitized_loader)
    defender_robust.robust_training(epochs=15)

    robust_acc, robust_cm, robust_report = evaluate_model(robust_model, test_loader)
    print(f"\nRobust Model Accuracy: {robust_acc:.4f}")
    print(robust_report)

    robust_model.eval()
    target_count_robust = 0

    with torch.no_grad():
        for inputs, labels in triggered_loader:
            inputs = inputs.to(device)
            outputs = robust_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            target_count_robust += (predicted.cpu() == target_class).sum().item()

    robust_backdoor_rate = target_count_robust / total
    print(f"Backdoor success rate against robust model: {robust_backdoor_rate:.4f}")
    print(
        f"Reduction in backdoor effectiveness: {(backdoor_success_rate - robust_backdoor_rate) / backdoor_success_rate:.2%}")

    print("\n=== Generating Results ===")
    os.makedirs("results/poisoning", exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(clean_losses, label='Clean Training')
    plt.plot(poisoned_losses, label='Poisoned Training')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(clean_accs, label='Clean Model')
    plt.plot(poisoned_accs, label='Poisoned Model')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('results/poisoning/training_curves.png')
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.heatmap(clean_cm, annot=True, fmt='d')
    plt.title(f'Clean Model\nAccuracy: {clean_acc:.4f}')

    plt.subplot(1, 3, 2)
    sns.heatmap(poisoned_cm, annot=True, fmt='d')
    plt.title(f'Poisoned Model\nAccuracy: {poisoned_acc:.4f}')

    plt.subplot(1, 3, 3)
    sns.heatmap(robust_cm, annot=True, fmt='d')
    plt.title(f'Robust Model\nAccuracy: {robust_acc:.4f}')
    plt.savefig('results/poisoning/confusion_matrices.png')
    plt.close()

    results = {
        'clean_accuracy': clean_acc,
        'poisoned_accuracy': poisoned_acc,
        'robust_accuracy': robust_acc,
        'backdoor_success_rate': backdoor_success_rate,
        'robust_backdoor_rate': robust_backdoor_rate,
        'defense_efficacy': (backdoor_success_rate - robust_backdoor_rate) / backdoor_success_rate,
        'outlier_detection_rate': true_positives / len(flipped_indices) if len(flipped_indices) > 0 else 0
    }

    pd.DataFrame.from_dict(results, orient='index', columns=['Value']).to_csv('results/poisoning/metrics.csv')

    with open('results/poisoning/reports.txt', 'w') as f:
        f.write("=== Clean Model ===\n")
        f.write(clean_report)
        f.write("\n\n=== Poisoned Model ===\n")
        f.write(poisoned_report)
        f.write("\n\n=== Robust Model ===\n")
        f.write(robust_report)
        f.write("\n\n=== Backdoor Attack Results ===\n")
        f.write(f"Original success rate: {backdoor_success_rate:.4f}\n")
        f.write(f"After defense: {robust_backdoor_rate:.4f}\n")
        f.write(f"Reduction: {(backdoor_success_rate - robust_backdoor_rate) / backdoor_success_rate:.2%}\n")


if __name__ == "__main__":
    main()
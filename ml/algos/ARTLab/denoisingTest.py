import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

NATURAL_DIR = 'perturbed_dataset/normal'
POISONED_DIR = 'perturbed_dataset'
PROCESSED_DIR = 'processed_dataset'
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f"[*] GPU Name: {torch.cuda.get_device_name(DEVICE)}")
else:
    print("[*] Using CPU")

output_dir = 'results/P3'
os.makedirs(output_dir, exist_ok=True)

class ComparisonDataset(Dataset):
    def __init__(self, natural_dir, poisoned_dirs, processed_dir, transform=None):
        self.natural_images = []
        for f in os.listdir(natural_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(natural_dir, f)
                processed_path = os.path.join(processed_dir, 'normal', f)
                if os.path.exists(processed_path):
                    self.natural_images.append(img_path)

        self.poisoned_images = []
        for d in poisoned_dirs:
            for f in os.listdir(d):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(d, f)
                    rel_path = os.path.relpath(img_path, POISONED_DIR)
                    processed_path = os.path.join(processed_dir, rel_path)
                    if os.path.exists(processed_path):
                        self.poisoned_images.append(img_path)

        self.processed_dir = processed_dir
        self.transform = transform

        print(f"\nImages naturelles valides: {len(self.natural_images)}/{len(os.listdir(natural_dir))}")
        print(f"Images empoisonnées valides: {len(self.poisoned_images)}/{sum(len(os.listdir(d)) for d in poisoned_dirs)}")

    def __len__(self):
        return len(self.natural_images) + len(self.poisoned_images)

    def __getitem__(self, idx):
        if idx < len(self.natural_images):
            img_path = self.natural_images[idx]
            label = 0
            processed_img_path = os.path.join(
                self.processed_dir,
                'normal',
                os.path.basename(img_path)
            )
        else:
            img_path = self.poisoned_images[idx - len(self.natural_images)]
            label = 1
            rel_path = os.path.relpath(img_path, POISONED_DIR)
            processed_img_path = os.path.join(
                self.processed_dir,
                rel_path
            )

        original_img = Image.open(img_path).convert('L')
        processed_img = Image.open(processed_img_path).convert('L')

        if self.transform:
            original_img = self.transform(original_img)
            processed_img = self.transform(processed_img)

        return {
            'original': original_img,
            'processed': processed_img,
            'label': label,
            'path': img_path
        }

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

poisoned_dirs = [os.path.join(POISONED_DIR, d) for d in os.listdir(POISONED_DIR)
                 if os.path.isdir(os.path.join(POISONED_DIR, d)) and d != 'normal']

print("Création du dataset...")
dataset = ComparisonDataset(NATURAL_DIR, poisoned_dirs, PROCESSED_DIR, transform=transform)
print(f"\nDataset créé avec {len(dataset)} échantillons valides")

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AttackDetector(nn.Module):
    def __init__(self):
        super(AttackDetector, self).__init__()

        self.original_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.processed_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, original, processed):
        orig_features = self.original_branch(original)
        proc_features = self.processed_branch(processed)
        combined = torch.cat([orig_features, proc_features], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        return output

class DenoisedAttackDetector(nn.Module):
    def __init__(self):
        super(DenoisedAttackDetector, self).__init__()

        self.original_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.processed_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.denoised_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, original, processed, denoised):
        orig_features = self.original_branch(original)
        proc_features = self.processed_branch(processed)
        denoised_features = self.denoised_branch(denoised)
        combined = torch.cat([orig_features, proc_features, denoised_features], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        return output

def train_denoising_autoencoder(model, dataloader, criterion, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Denoising Epoch {epoch+1}/{epochs}"):
            images = batch['original'].to(device)
            noisy_images = images + 0.1 * torch.randn_like(images)
            noisy_images = torch.clamp(noisy_images, 0., 1.)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

def apply_denoising(model, dataloader, device):
    model.eval()
    denoised_images = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Applying denoising"):
            images = batch['original'].to(device)
            denoised = model(images)
            denoised_images.append(denoised.cpu())
    return torch.cat(denoised_images, dim=0)

def train_epoch(model, dataloader, criterion, optimizer, device, use_denoising=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    for batch in tqdm(dataloader, desc="Training"):
        original = batch['original'].to(device)
        processed = batch['processed'].to(device)
        labels = batch['label'].float().to(device).view(-1, 1)

        if use_denoising:
            denoised = batch['denoised'].to(device)
            outputs = model(original, processed, denoised)
        else:
            outputs = model(original, processed)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().detach().numpy())
        all_preds.extend(predicted.cpu().detach().numpy())
        all_probs.extend(outputs.cpu().detach().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)

    return epoch_loss, epoch_acc, roc_auc, pr_auc, all_labels, all_preds, all_probs

def evaluate(model, dataloader, criterion, device, use_denoising=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            original = batch['original'].to(device)
            processed = batch['processed'].to(device)
            labels = batch['label'].float().to(device).view(-1, 1)

            if use_denoising:
                denoised = batch['denoised'].to(device)
                outputs = model(original, processed, denoised)
            else:
                outputs = model(original, processed)

            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)

    return epoch_loss, epoch_acc, auc_score, pr_auc, all_labels, all_preds, fpr, tpr, precision, recall

def get_classification_report(labels, preds, target_names=['Naturel', 'Attaqué']):
    try:
        report = classification_report(labels, preds, target_names=target_names, output_dict=True, zero_division=0)
        return report
    except Exception as e:
        print(f"Erreur lors de la génération du rapport: {e}")
        return {
            'Naturel': {'precision': 0, 'recall': 0, 'f1-score': 0},
            'Attaqué': {'precision': 0, 'recall': 0, 'f1-score': 0},
            'accuracy': 0,
            'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0},
            'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0}
        }

if __name__ == '__main__':
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    denoising_model = DenoisingAutoencoder().to(DEVICE)
    denoising_criterion = nn.MSELoss()
    denoising_optimizer = optim.Adam(denoising_model.parameters(), lr=0.001)

    print("\nEntraînement de l'autoencodeur de débroitage...")
    train_denoising_autoencoder(denoising_model, train_loader, denoising_criterion, denoising_optimizer, DEVICE)

    print("\nApplication du débroitage aux images...")
    denoised_images = apply_denoising(denoising_model, train_loader, DEVICE)

    class DenoisedDataset(Dataset):
        def __init__(self, original_dataset, denoised_images):
            self.original_dataset = original_dataset
            self.denoised_images = denoised_images

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            item = self.original_dataset[idx]
            denoised_img = self.denoised_images[idx]
            return {
                'original': item['original'],
                'processed': item['processed'],
                'denoised': denoised_img,
                'label': item['label'],
                'path': item['path']
            }

    denoised_train_dataset = DenoisedDataset(train_dataset, denoised_images)
    denoised_train_loader = DataLoader(denoised_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    denoised_val_dataset = DenoisedDataset(val_dataset, denoised_images[:len(val_dataset)])
    denoised_val_loader = DataLoader(denoised_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    denoised_test_dataset = DenoisedDataset(test_dataset, denoised_images[:len(test_dataset)])
    denoised_test_loader = DataLoader(denoised_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\nDébut de l'entraînement avec débroitage...")
    denoised_model = DenoisedAttackDetector().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(denoised_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_aucs = []
    val_pr_aucs = []
    train_metrics_per_class = []
    val_metrics_per_class = []

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_auc, train_pr_auc, train_labels, train_preds, train_probs = train_epoch(
            denoised_model, denoised_train_loader, criterion, optimizer, DEVICE, use_denoising=True)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc, val_auc, val_pr_auc, val_labels, val_preds, fpr, tpr, precision, recall = evaluate(
            denoised_model, denoised_val_loader, criterion, DEVICE, use_denoising=True)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        val_pr_aucs.append(val_pr_auc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(denoised_model.state_dict(), 'best_denoised_attack_detector.pth')

        train_report = get_classification_report(train_labels, train_preds)
        val_report = get_classification_report(val_labels, val_preds)

        train_metrics_per_class.append({
            'epoch': epoch+1,
            'natural_precision': train_report['Naturel']['precision'],
            'natural_recall': train_report['Naturel']['recall'],
            'natural_f1': train_report['Naturel']['f1-score'],
            'attacked_precision': train_report['Attaqué']['precision'],
            'attacked_recall': train_report['Attaqué']['recall'],
            'attacked_f1': train_report['Attaqué']['f1-score']
        })

        val_metrics_per_class.append({
            'epoch': epoch+1,
            'natural_precision': val_report['Naturel']['precision'],
            'natural_recall': val_report['Naturel']['recall'],
            'natural_f1': val_report['Naturel']['f1-score'],
            'attacked_precision': val_report['Attaqué']['precision'],
            'attacked_recall': val_report['Attaqué']['recall'],
            'attacked_f1': val_report['Attaqué']['f1-score']
        })

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Train PR AUC: {train_pr_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val PR AUC: {val_pr_auc:.4f}")

    print("\nChargement du meilleur modèle avec débroitage pour l'évaluation finale...")
    denoised_model.load_state_dict(torch.load('best_denoised_attack_detector.pth'))

    test_loss, test_acc, test_auc, test_pr_auc, test_labels, test_preds, fpr, tpr, precision, recall = evaluate(
        denoised_model, denoised_test_loader, criterion, DEVICE, use_denoising=True)

    print("\nRésultats finaux sur le test (modèle avec débroitage):")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test PR AUC: {test_pr_auc:.4f}")

    metrics_df = pd.DataFrame({
        'Epoch': range(1, EPOCHS+1),
        'Train Loss': train_losses,
        'Val Loss': val_losses,
        'Train Acc': train_accs,
        'Val Acc': val_accs,
        'Val AUC': val_aucs,
        'Val PR AUC': val_pr_aucs,
        'Natural Precision': [m['natural_precision'] for m in val_metrics_per_class],
        'Natural Recall': [m['natural_recall'] for m in val_metrics_per_class],
        'Natural F1': [m['natural_f1'] for m in val_metrics_per_class],
        'Attacked Precision': [m['attacked_precision'] for m in val_metrics_per_class],
        'Attacked Recall': [m['attacked_recall'] for m in val_metrics_per_class],
        'Attacked F1': [m['attacked_f1'] for m in val_metrics_per_class]
    })

    metrics_df.to_csv(os.path.join(output_dir, 'denoised_training_metrics.csv'), index=False)
    print(f"\nMétriques sauvegardées dans {os.path.join(output_dir, 'denoised_training_metrics.csv')}")

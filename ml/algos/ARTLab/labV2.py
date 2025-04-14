import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer

# Decorator pour mes analyses de temps
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"[TIME] {method.__name__} executed in {(te - ts):.2f} seconds")
        return result, te - ts
    return timed

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {device}")

# Verif gpu parce que mskn sur le processeur ça prend 20 ans
if torch.cuda.is_available():
    print(f"[*] GPU Name: {torch.cuda.get_device_name(device)}")
    print(f"[*] CUDA Version: {torch.version.cuda}")
    print(f"[*] GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.2f} GB")

# classe pour adapter au dataset
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

# Data preparation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@timeit
def prepare_data():
    dataset = BrainMRIDataset(csv_file='adni_dataset/train.csv',
                              root_dir='adni_dataset/ADNI_IMAGES/png_images',
                              transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset

(train_loader, test_loader, train_dataset, test_dataset), prep_time = prepare_data()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 5)
        self.dropout = nn.Dropout(0.5)  # Added for better regularization

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

@timeit
def train_model(model, train_loader, epochs=15):
    train_losses = []
    model.train()

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
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
        epoch_time = time.time() - epoch_start

        print(f"[+] Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Time: {epoch_time:.2f}s")

    return train_losses

print("[*] Training initial model...")
train_losses, train_time = train_model(model, train_loader)

os.makedirs("img", exist_ok=True)
os.makedirs("results", exist_ok=True)

# visu
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['Training Time'], [train_time], color='blue')
plt.ylabel('Time (seconds)')
plt.title('Training Execution Time')
plt.tight_layout()
plt.savefig('img/training_metrics.png')

# fonction d'évaluation
@timeit
def evaluate_model(model, dataloader, device, attack_name=None):
    model.eval()
    all_labels = []
    all_preds = []
    total_time = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            batch_time = time.time() - start_time
            total_time += batch_time

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    cm = confusion_matrix(all_labels, all_preds)
    avg_inference_time = total_time / len(dataloader)

    if attack_name:
        print(f"\n[*] Evaluation under {attack_name} attack:")
    else:
        print("\n[*] Clean evaluation:")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    return accuracy, cm, avg_inference_time

(clean_acc, clean_cm, clean_time), eval_time = evaluate_model(model, test_loader, device)

# les matriceuuuuu
plt.figure(figsize=(8, 6))
sns.heatmap(clean_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'],
            yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'])
plt.title('Initial Confusion Matrix\nAccuracy: {:.2f}%'.format(clean_acc * 100))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('img/initial_confusion_matrix.png')


art_classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 128, 128),
    nb_classes=5,
    clip_values=(0, 1)
)

# test des atk amélioré
@timeit
def test_evasion_attack(attack, name, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    total_time = 0

    for inputs, labels in test_loader:
        attack_start = time.time()
        inputs, labels = inputs.to(device), labels.to(device)

        # generation exemples
        x_adv = attack.generate(inputs.cpu().numpy())
        adv_inputs = torch.FloatTensor(x_adv).to(device)

        # inference
        with torch.no_grad():
            outputs = model(adv_inputs)
            _, preds = torch.max(outputs, 1)

        batch_time = time.time() - attack_start
        total_time += batch_time

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    cm = confusion_matrix(all_labels, all_preds)
    avg_time = total_time / len(test_loader)

    print(f"\n[*] Attack: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average attack+inference time per batch: {avg_time:.4f} seconds")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    return accuracy, cm, avg_time

print("\n=== Evasion Attacks ===")
fgsm = FastGradientMethod(art_classifier, eps=0.7)
pgd = ProjectedGradientDescent(art_classifier, eps=0.7, max_iter=10)

(fgsm_acc, fgsm_cm, fgsm_time), fgsm_eval_time = test_evasion_attack(fgsm, "FGSM (ε=0.7)", test_loader, device)
(pgd_acc, pgd_cm, pgd_time), pgd_eval_time = test_evasion_attack(pgd, "PGD (ε=0.7, iter=10)", test_loader, device)

# visu des atk
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(clean_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Clean Accuracy: {:.2f}%'.format(clean_acc * 100))
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 3, 2)
sns.heatmap(fgsm_cm, annot=True, fmt='d', cmap='Reds')
plt.title('FGSM Attack Accuracy: {:.2f}%'.format(fgsm_acc * 100))
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 3, 3)
sns.heatmap(pgd_cm, annot=True, fmt='d', cmap='Reds')
plt.title('PGD Attack Accuracy: {:.2f}%'.format(pgd_acc * 100))
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.savefig('img/attack_comparison.png')

# comparaison perf.
attack_metrics = {
    'Clean': {'accuracy': clean_acc, 'time': clean_time},
    'FGSM': {'accuracy': fgsm_acc, 'time': fgsm_time},
    'PGD': {'accuracy': pgd_acc, 'time': pgd_time}
}

# entrainement adversaire - techno de protection simple
print("\n=== Adversarial Training ===")

@timeit
def adversarial_training(art_classifier, train_dataset, attacks, epochs=15):
    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))]).numpy()
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))]).numpy()

    trainer = AdversarialTrainer(art_classifier, attacks=attacks, ratio=0.5)
    trainer.fit(X_train, y_train, nb_epochs=epochs)

    return trainer

trainer, adv_train_time = adversarial_training(
    art_classifier,
    train_dataset,
    attacks=[fgsm, pgd]
)


@timeit
def visualize_attacks(model, test_loader, device, art_classifier, num_examples=5):
    model.eval()
    os.makedirs("img/attacks", exist_ok=True)

    inputs, labels = next(iter(test_loader))
    inputs, labels = inputs.to(device), labels.to(device)

    fgsm = FastGradientMethod(art_classifier, eps=0.7)
    pgd = ProjectedGradientDescent(art_classifier, eps=0.7, max_iter=10)

    x_adv_fgsm = torch.FloatTensor(fgsm.generate(inputs.cpu().numpy())).to(device)
    x_adv_pgd = torch.FloatTensor(pgd.generate(inputs.cpu().numpy())).to(device)

    indices = np.random.choice(len(inputs), num_examples, replace=False)

    plt.figure(figsize=(15, 5 * num_examples))

    for i, idx in enumerate(indices):
        original_img = inputs[idx].cpu().squeeze().numpy()
        fgsm_img = x_adv_fgsm[idx].cpu().squeeze().numpy()
        pgd_img = x_adv_pgd[idx].cpu().squeeze().numpy()
        fgsm_diff = np.abs(original_img - fgsm_img)
        pgd_diff = np.abs(original_img - pgd_img)

        plt.subplot(num_examples, 5, i * 5 + 1)
        plt.imshow(original_img, cmap='gray')
        plt.title(f"Original (Label: {labels[idx].item()})")
        plt.axis('off')

        plt.subplot(num_examples, 5, i * 5 + 2)
        plt.imshow(fgsm_img, cmap='gray')
        plt.title("FGSM Perturbed")
        plt.axis('off')

        plt.subplot(num_examples, 5, i * 5 + 3)
        plt.imshow(fgsm_diff, cmap='hot')
        plt.title("FGSM Difference")
        plt.axis('off')

        plt.subplot(num_examples, 5, i * 5 + 4)
        plt.imshow(pgd_img, cmap='gray')
        plt.title("PGD Perturbed")
        plt.axis('off')

        plt.subplot(num_examples, 5, i * 5 + 5)
        plt.imshow(pgd_diff, cmap='hot')
        plt.title("PGD Difference")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('img/attacks/attack_visualization.png')
    plt.close()

    for i, idx in enumerate(indices):
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))

        original_img = inputs[idx].cpu().squeeze().numpy()
        fgsm_img = x_adv_fgsm[idx].cpu().squeeze().numpy()
        pgd_img = x_adv_pgd[idx].cpu().squeeze().numpy()

        fgsm_diff = np.abs(original_img - fgsm_img)
        pgd_diff = np.abs(original_img - pgd_img)

        axs[0].imshow(original_img, cmap='gray')
        axs[0].set_title(f"Original (Label: {labels[idx].item()})")
        axs[0].axis('off')

        axs[1].imshow(fgsm_img, cmap='gray')
        axs[1].set_title("FGSM Perturbed")
        axs[1].axis('off')

        axs[2].imshow(fgsm_diff, cmap='hot')
        axs[2].set_title("FGSM Difference")
        axs[2].axis('off')

        axs[3].imshow(pgd_img, cmap='gray')
        axs[3].set_title("PGD Perturbed")
        axs[3].axis('off')

        axs[4].imshow(pgd_diff, cmap='hot')
        axs[4].set_title("PGD Difference")
        axs[4].axis('off')

        plt.tight_layout()
        plt.savefig(f'img/attacks/attack_example_{i}.png')
        plt.close()

# eval post-défense
(final_acc, final_cm, final_time), final_eval_time = evaluate_model(model, test_loader, device, "after defense")

(fgsm_acc_def, fgsm_cm_def, fgsm_time_def), fgsm_eval_time_def = test_evasion_attack(
    fgsm, "FGSM (after defense)", test_loader, device
)

(pgd_acc_def, pgd_cm_def, pgd_time_def), pgd_eval_time_def = test_evasion_attack(
    pgd, "PGD (after defense)", test_loader, device
)

# comparaison de l'efficacité
defense_metrics = {
    'Before Defense': {
        'Clean': clean_acc,
        'FGSM': fgsm_acc,
        'PGD': pgd_acc,
        'Training Time': train_time
    },
    'After Defense': {
        'Clean': final_acc,
        'FGSM': fgsm_acc_def,
        'PGD': pgd_acc_def,
        'Training Time': adv_train_time
    }
}

# visualisation de l'efficacité des défenses
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.bar(defense_metrics['Before Defense'].keys(),
        defense_metrics['Before Defense'].values(),
        color=['blue', 'red', 'red', 'green'])
plt.title('Performance Before Defense')
plt.ylabel('Accuracy / Time (s)')
plt.xticks(rotation=45)
plt.ylim(0, 1.2)

plt.subplot(1, 3, 2)
plt.bar(defense_metrics['After Defense'].keys(),
        defense_metrics['After Defense'].values(),
        color=['blue', 'red', 'red', 'green'])
plt.title('Performance After Defense')
plt.ylabel('Accuracy / Time (s)')
plt.xticks(rotation=45)
plt.ylim(0, 1.2)

plt.subplot(1, 3, 3)
width = 0.35
x = np.arange(3)
plt.bar(x - width / 2,
        [defense_metrics['Before Defense']['Clean'],
         defense_metrics['Before Defense']['FGSM'],
         defense_metrics['Before Defense']['PGD']],
        width, label='Before Defense')
plt.bar(x + width / 2,
        [defense_metrics['After Defense']['Clean'],
         defense_metrics['After Defense']['FGSM'],
         defense_metrics['After Defense']['PGD']],
        width, label='After Defense')
plt.xticks(x, ['Clean', 'FGSM', 'PGD'])
plt.ylabel('Accuracy')
plt.title('Defense Effectiveness Comparison')
plt.legend()

plt.tight_layout()
plt.savefig('img/defense_comparison.png')

# sauvegarde des val. pour comparaison
metrics_df = pd.DataFrame({
    'Scenario': ['Clean', 'FGSM Attack', 'PGD Attack',
                 'Clean (After Defense)', 'FGSM (After Defense)', 'PGD (After Defense)'],
    'Accuracy': [clean_acc, fgsm_acc, pgd_acc,
                 final_acc, fgsm_acc_def, pgd_acc_def],
    'Inference Time': [clean_time, fgsm_time, pgd_time,
                       final_time, fgsm_time_def, pgd_time_def]
})

metrics_df.to_csv('results/metrics_comparison.csv', index=False)

# comp. finale
plt.figure(figsize=(8, 5))
plt.bar(['Standard Training', 'Adversarial Training'],
        [train_time, adv_train_time],
        color=['blue', 'orange'])
plt.ylabel('Time (seconds)')
plt.title('Training Time Comparison')
plt.savefig('img/training_time_comparison.png')

print("\n=== Final Summary ===")
print("\nAccuracy Metrics:")
print(f"Initial clean accuracy: {clean_acc:.4f}")
print(f"Accuracy under FGSM attack: {fgsm_acc:.4f} (Drop: {(clean_acc - fgsm_acc):.4f})")
print(f"Accuracy under PGD attack: {pgd_acc:.4f} (Drop: {(clean_acc - pgd_acc):.4f})")
print(f"Clean accuracy after defense: {final_acc:.4f}")
print(f"Accuracy under FGSM after defense: {fgsm_acc_def:.4f} (Improvement: {(fgsm_acc_def - fgsm_acc):.4f})")
print(f"Accuracy under PGD after defense: {pgd_acc_def:.4f} (Improvement: {(pgd_acc_def - pgd_acc):.4f})")

print("\nPerformance Metrics:")
print(f"Standard training time: {train_time:.2f} seconds")
print(f"Adversarial training time: {adv_train_time:.2f} seconds ({(adv_train_time / train_time - 1) * 100:.2f}% increase)")
print(f"Average clean inference time: {clean_time:.4f} seconds per batch")
print(f"Average FGSM attack+inference time: {fgsm_time:.4f} seconds per batch")
print(f"Average PGD attack+inference time: {pgd_time:.4f} seconds per batch")

# generation rapport
with open('results/txt/final_report.txt', 'w') as f:
    f.write("=== Adversarial Robustness Experiment Report ===\n\n")
    f.write("Key Findings:\n")
    f.write(f"- The model's accuracy drops from {clean_acc:.2%} to {fgsm_acc:.2%} under FGSM attack ({((clean_acc - fgsm_acc) / clean_acc):.2%} reduction)\n")
    f.write(f"- Under more sophisticated PGD attack, accuracy drops further to {pgd_acc:.2%}\n")
    f.write(f"- Adversarial training improves robustness, reducing FGSM effectiveness by {((fgsm_acc_def - fgsm_acc) / fgsm_acc):.2%}\n")
    f.write(f"- The trade-off is a {((adv_train_time / train_time - 1) * 100):.2f}% increase in training time\n\n")

print("\n[*] Generating attack visualizations...")
visualize_attacks(model, test_loader, device, art_classifier)
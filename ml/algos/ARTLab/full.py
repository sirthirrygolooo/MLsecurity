import os

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    from art.defences.trainer import AdversarialTrainer
except ImportError:
    print('Import error')
    exit(1)


torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {device}")

if torch.cuda.is_available():
    print(f"[*] GPU Name: {torch.cuda.get_device_name(device)}")


# dataset custom
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

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = BrainMRIDataset(csv_file='adni_dataset/train.csv',
                          root_dir='adni_dataset/ADNI_IMAGES/png_images',
                          transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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

print("[*] Training initial model...")
train_losses = []
for epoch in range(10):
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
    print(f"[+] Epoch {epoch + 1}/10 - Loss: {epoch_loss:.4f}")

# setup dir de sortie
os.makedirs("img", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.savefig('img/training_loss.png')


def evaluate_model(model, dataloader, device):
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

    return accuracy, cm


clean_acc, clean_cm = evaluate_model(model, test_loader, device)
print(f"\n[*] Initial accuracy: {clean_acc:.4f}")

# matrices de confusion res modl
plt.figure(figsize=(8, 6))
sns.heatmap(clean_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Initial Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('img/initial_confusion_matrix.png')

# art config
art_classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 128, 128),
    nb_classes=5,
    clip_values=(0, 1)
)


# attaques par evasion
def test_evasion_attack(attack, name, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        x_adv = attack.generate(inputs.cpu().numpy())
        adv_inputs = torch.FloatTensor(x_adv).to(device)

        # predictions
        with torch.no_grad():
            outputs = model(adv_inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    cm = confusion_matrix(all_labels, all_preds)

    print(f"[*] Accuracy after {name}: {accuracy:.4f}")
    return accuracy, cm


print("\n=== Evasion Attacks ===")
fgsm = FastGradientMethod(art_classifier, eps=0.2)
pgd = ProjectedGradientDescent(art_classifier, eps=0.2, max_iter=10)

fgsm_acc, fgsm_cm = test_evasion_attack(fgsm, "FGSM", test_loader, device)
pgd_acc, pgd_cm = test_evasion_attack(pgd, "PGD", test_loader, device)

# matrices de confusion atk
plt.figure(figsize=(8, 6))
sns.heatmap(fgsm_cm, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix after FGSM Attack')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('img/fgsm_confusion_matrix.png')
plt.figure(figsize=(8, 6))
sns.heatmap(pgd_cm, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix after PGD Attack')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('img/pgd_confusion_matrix.png')


print("\n=== Adversarial Training ===")

# preparation datas
X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))]).numpy()
y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))]).numpy()

trainer = AdversarialTrainer(art_classifier, attacks=[fgsm, pgd], ratio=0.5)

trainer.fit(X_train, y_train, nb_epochs=15)

# eval finale
final_acc, final_cm = evaluate_model(model, test_loader, device)
print(f"\n[*] Accuracy after adversarial training: {final_acc:.4f}")

# res après defense
fgsm_acc_def, fgsm_cm_def = test_evasion_attack(fgsm, "FGSM (after defense)", test_loader, device)
pgd_acc_def, pgd_cm_def = test_evasion_attack(pgd, "PGD (after defense)", test_loader, device)

# matrices de fonfusion
plt.figure(figsize=(8, 6))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix After Adversarial Training')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('img/final_confusion_matrix.png')

plt.figure(figsize=(8, 6))
sns.heatmap(fgsm_cm_def, annot=True, fmt='d', cmap='Purples')
plt.title('Confusion Matrix After Defense (FGSM)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('img/defense_fgsm_confusion_matrix.png')

plt.figure(figsize=(8, 6))
sns.heatmap(pgd_cm_def, annot=True, fmt='d', cmap='Purples')
plt.title('Confusion Matrix After Defense (PGD)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('img/defense_pgd_confusion_matrix.png')

print("\n=== Summary ===")
print(f"Initial accuracy: {clean_acc:.4f}")
print(f"Accuracy after FGSM attack: {fgsm_acc:.4f}")
print(f"Accuracy after PGD attack: {pgd_acc:.4f}")
print(f"Accuracy after adversarial training: {final_acc:.4f}")
print(f"Accuracy against FGSM after defense: {fgsm_acc_def:.4f}")
print(f"Accuracy against PGD after defense: {pgd_acc_def:.4f}")
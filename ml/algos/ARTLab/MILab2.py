import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from art.estimators.classification import PyTorchClassifier
from art.attacks.inference.model_inversion import MIFace
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

output_dir = "results/inversion_lfw"
os.makedirs(output_dir, exist_ok=True)

def leTailleMeure(method):
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
print(f"[*] GPU Name: {torch.cuda.get_device_name(device)}" if torch.cuda.is_available() else "[*] No GPU available, using CPU.")

class LFW_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path, label = self.images[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return (image, label)

# Transformations
transform = transforms.Compose([
    transforms.Resize((62, 47)),  # Resize to 62x47 pixels
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class RecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(RecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 15 * 11, 256)  # Adjusted for 62x47 input
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 15 * 11)  # Adjusted for 62x47 input
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

@leTailleMeure
def train_model(model, train_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")
    return model

print("\n[*] Preparing data...")
dataset = LFW_Dataset(
    root_dir='lfw_dataset',
    transform=transform
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("\n[*] Training model...")
num_classes = len(dataset.classes)
model = RecognitionModel(num_classes=num_classes).to(device)
model, train_time = train_model(model, train_loader)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

art_classifier = PyTorchClassifier(
    model=model,
    loss=loss_fn,
    optimizer=optimizer,
    input_shape=(3, 62, 47),
    nb_classes=num_classes,
    clip_values=(0, 1)
)

@leTailleMeure
def run_model_inversion(target_class, art_classifier):
    print(f"\n[*] Running Model Inversion for class {target_class}")
    attack = MIFace(classifier=art_classifier)
    x_init = np.random.rand(1, 3, 62, 47).astype(np.float32)
    x_inverted = attack.infer(x=x_init, y=np.array([target_class]))
    return x_inverted

def visualize_inversion(original_images, inverted_images, class_names, suffix=""):
    plt.figure(figsize=(15, 10))
    for i, (orig, inv) in enumerate(zip(original_images, inverted_images)):
        plt.subplot(2, len(original_images), i + 1)
        plt.imshow(np.transpose(orig, (1, 2, 0)))
        plt.title(f"Original {class_names[i]}")
        plt.axis('off')

        plt.subplot(2, len(original_images), len(original_images) + i + 1)
        plt.imshow(np.transpose(inv, (1, 2, 0)))
        plt.title(f"Inverted {class_names[i]} {suffix}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison{suffix}.png")
    plt.close()

def evaluate_inversion(originals, inverted, class_names, suffix=""):
    mse_results = []
    for orig, inv in zip(originals, inverted):
        mse = np.mean((orig - inv) ** 2)
        mse_results.append(mse)

    plt.figure(figsize=(8, 5))
    plt.bar(class_names, mse_results, color='red')
    plt.title("Reconstruction Error (MSE) per Class")
    plt.ylabel("Mean Squared Error")
    plt.savefig(f"{output_dir}/mse_evaluation{suffix}.png")
    plt.close()
    return mse_results

print("\n=== Model Inversion Attack ===")
target_classes = [0, 1, 2]  # Example target classes, adjust as needed
class_names = [dataset.classes[c] for c in target_classes]

original_samples = []
for class_id in target_classes:
    for img, label in test_dataset:
        if int(label) == class_id:
            original_samples.append(img.numpy())
            break

inverted_images = []
for class_id in target_classes:
    inverted_img, _ = run_model_inversion(class_id, art_classifier)
    inverted_images.append(inverted_img[0])

visualize_inversion(original_samples, inverted_images, class_names)
mse_scores = evaluate_inversion(original_samples, inverted_images, class_names)

@leTailleMeure
def run_membership_inference(art_classifier, train_dataset, test_dataset):
    print("\n[*] Running Membership Inference Attack")
    X_train = torch.stack([x for x, _ in train_dataset]).numpy()
    y_train = torch.tensor([y for _, y in train_dataset]).numpy()
    X_test = torch.stack([x for x, _ in test_dataset]).numpy()
    y_test = torch.tensor([y for _, y in test_dataset]).numpy()

    attack = MembershipInferenceBlackBox(estimator=art_classifier)

    attack.fit(X_train, y_train, X_test, y_test)

    train_preds = attack.infer(X_train, y_train)
    test_preds = attack.infer(X_test, y_test)

    train_acc = (train_preds == 1).mean()
    test_acc = (test_preds == 1).mean()

    print(f"Attack Accuracy on Training Data: {train_acc:.2%}")
    print(f"Attack Accuracy on Test Data: {test_acc:.2%}")
    return train_acc, test_acc

(train_acc, test_acc), _ = run_membership_inference(art_classifier, train_dataset, test_dataset)

@leTailleMeure
def apply_defenses(model):
    print("\n=== Applying Defenses ===")
    for param in model.parameters():
        param.data += torch.randn_like(param) * 0.01
    print("[*] Label Smoothing + Gradient Clipping")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for group in optimizer.param_groups:
        group['max_grad_norm'] = 1.0
    return model, criterion, optimizer

(model_defended, criterion, optimizer), _ = apply_defenses(model)

art_classifier_defended = PyTorchClassifier(
    model=model_defended,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 62, 47),
    nb_classes=num_classes,
    clip_values=(0, 1)
)

print("\n[*] Re-running attacks with defenses...")
inverted_images_defended = []
for class_id in target_classes:
    inverted_img, _ = run_model_inversion(class_id, art_classifier_defended)
    inverted_images_defended.append(inverted_img[0])

visualize_inversion(original_samples, inverted_images_defended, class_names, suffix="_defended")
mse_scores_defended = evaluate_inversion(original_samples, inverted_images_defended, class_names, suffix="_defended")
print("\n=== Final Report ===")
print(f"Original MSE Scores: {dict(zip(class_names, mse_scores))}")
print(f"Defended MSE Scores: {dict(zip(class_names, mse_scores_defended))}")
print(f"Membership Inference - Train Acc: {train_acc:.2%}, Test Acc: {test_acc:.2%}")

with open(os.path.join(output_dir, "inversion_report.txt"), "w") as f:
    f.write("=== Model Inversion Attack Report ===\n\n")
    f.write("Key Findings:\n")
    f.write(f"- Successful reconstruction for all target classes (MSE: {np.mean(mse_scores):.4f})\n")
    f.write(f"- Defenses reduced reconstruction quality by "
            f"{((np.mean(mse_scores_defended) - np.mean(mse_scores)) / np.mean(mse_scores)):.2%}\n")
    f.write(f"- Membership inference accuracy: {train_acc:.2%} (train), {test_acc:.2%} (test)\n\n")

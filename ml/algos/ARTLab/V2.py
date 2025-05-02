import os
import time
import csv
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

EPSILON = 0.02

def setup_environment():
    """Setup the environment by checking dataset and renaming images if necessary."""
    if not os.path.exists('adni_dataset2/AugmentedAlzheimerDataset/AD/AD-0001.jpg'):
        if not os.path.exists('adni_dataset2'):
            print('Dataset Missing, run setup.py ? [y/N]')
            choice = input()
            if choice.lower() == 'y':
                os.system('python setup.py')
                print('Fin de setup.py\n')
                print(f"\n Executing {__file__}")
                print("\n ")
            else:
                print("Dataset not found. Exiting.")
                exit()
        print("[*] Renaming images...")
        rename_images_in_directory('adni_dataset2/AugmentedAlzheimerDataset/AD', 'AD')
        rename_images_in_directory('adni_dataset2/AugmentedAlzheimerDataset/CN', 'CN')
        rename_images_in_directory('adni_dataset2/AugmentedAlzheimerDataset/EMCI', 'EMCI')
        rename_images_in_directory('adni_dataset2/AugmentedAlzheimerDataset/LMCI', 'LMCI')

def rename_images_in_directory(directory_path, prefix):
    """Rename images in the directory with a given prefix."""
    files = sorted(os.listdir(directory_path))
    for counter, filename in enumerate(files, start=1):
        new_name = f"{prefix}-{counter:04d}{os.path.splitext(filename)[1]}"
        os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, new_name))

def create_csv_if_not_exists():
    """Create a CSV file from images if it doesn't exist."""
    if not os.path.exists('adni_dataset2/train.csv'):
        process_images_to_csv('adni_dataset2/AugmentedAlzheimerDataset/AD', 'adni_dataset2/train.csv')
        process_images_to_csv('adni_dataset2/AugmentedAlzheimerDataset/CN', 'adni_dataset2/train.csv')
        process_images_to_csv('adni_dataset2/AugmentedAlzheimerDataset/EMCI', 'adni_dataset2/train.csv')
        process_images_to_csv('adni_dataset2/AugmentedAlzheimerDataset/LMCI', 'adni_dataset2/train.csv')
        print(f"Les données ont été enregistrées dans adni_dataset2/train.csv")
        print("\n[ ] ------------------------------------------------------------")

def process_images_to_csv(directory_path, output_csv_path):
    """Process images in the directory and write to a CSV file."""
    with open(output_csv_path, mode='a', newline='') as csv_file:
        fieldnames = ['id_code', 'diagnosis']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_file.tell() == 0:
            writer.writeheader()
        for filename in sorted(os.listdir(directory_path)):
            name_without_extension = os.path.splitext(filename)[0]
            diagnosis = {'AD': 3, 'LMCI': 2, 'EMCI': 1, 'CN': 0}.get(name_without_extension.split('-')[0], -1)
            writer.writerow({'id_code': name_without_extension, 'diagnosis': diagnosis})

def shuffle_csv(input_csv):
    """Shuffle the CSV file."""
    df = pd.read_csv(input_csv)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(input_csv, index=False)

def timeit(method):
    """Decorator to measure the execution time of a function."""
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"[TIME] {method.__name__} executed in {(te - ts):.2f} seconds")
        return result, te - ts
    return timed

def set_device():
    """Set the device for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(device)}")
        print(f"[*] CUDA Version: {torch.version.cuda}")
        print(f"[*] GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.2f} GB")
    return device

class BrainMRIDataset(Dataset):
    """Dataset class for Brain MRI images."""
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        folder_name = img_name.split('-')[0]
        img_path = os.path.join(self.root_dir, folder_name, img_name + '.jpg')
        image = Image.open(img_path).convert('L')
        y_label = torch.tensor(self.annotations.iloc[index, 1])
        if self.transform:
            image = self.transform(image)
        return image, y_label

@timeit
def prepare_data(csv_file, root_dir, transform):
    """Prepare data loaders for training and testing."""
    dataset = BrainMRIDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset

class Net(nn.Module):
    """CNN model for classification."""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        dummy_input = torch.zeros(1, 1, 200, 190)
        dummy_output = self.pool(torch.relu(self.conv2(self.pool(torch.relu(self.conv1(dummy_input))))))
        flattened_size = dummy_output.view(-1).shape[0]
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

@timeit
def train_model(model, train_loader, criterion, optimizer, device, epochs=20):
    """Train the model."""
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

def plot_training_metrics(train_losses, train_time):
    """Plot training metrics."""
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

@timeit
def evaluate_model(model, dataloader, device, criterion, attack_name=None):
    """Evaluate the model."""
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

def plot_confusion_matrix(cm, accuracy, title, filename):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
                yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    plt.title(f'{title}\nAccuracy: {accuracy * 100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filename)

@timeit
def test_evasion_attack(attack, name, test_loader, device, model):
    """Test evasion attacks."""
    model.eval()
    all_labels = []
    all_preds = []
    total_time = 0
    for inputs, labels in test_loader:
        attack_start = time.time()
        inputs, labels = inputs.to(device), labels.to(device)
        x_adv = attack.generate(inputs.cpu().numpy())
        adv_inputs = torch.FloatTensor(x_adv).to(device)
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

def plot_attack_comparison(clean_cm, clean_acc, fgsm_cm, fgsm_acc, pgd_cm, pgd_acc):
    """Plot attack comparison."""
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    sns.heatmap(clean_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Clean Accuracy: {clean_acc * 100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.subplot(1, 3, 2)
    sns.heatmap(fgsm_cm, annot=True, fmt='d', cmap='Reds')
    plt.title(f'FGSM Attack Accuracy: {fgsm_acc * 100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.subplot(1, 3, 3)
    sns.heatmap(pgd_cm, annot=True, fmt='d', cmap='Reds')
    plt.title(f'PGD Attack Accuracy: {pgd_acc * 100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('img/attack_comparison.png')

def plot_performance_metrics(attack_metrics):
    """Plot performance metrics."""
    plt.figure(figsize=(12, 5))
    plt.bar(attack_metrics.keys(), [attack_metrics[key]['accuracy'] for key in attack_metrics], color=['blue', 'red', 'red'])
    plt.ylabel('Accuracy')
    plt.title('Accuracy Under Different Scenarios')
    plt.savefig('img/accuracy_comparison.png')
    plt.figure(figsize=(12, 5))
    plt.bar(attack_metrics.keys(), [attack_metrics[key]['time'] for key in attack_metrics], color=['blue', 'red', 'red'])
    plt.ylabel('Average Time per Batch (seconds)')
    plt.title('Inference Time Under Different Scenarios')
    plt.savefig('img/time_comparison.png')

def save_metrics(clean_acc, fgsm_acc, pgd_acc, clean_time, fgsm_time, pgd_time):
    """Save metrics to CSV."""
    metrics_df = pd.DataFrame({
        'Scenario': ['Clean', 'FGSM Attack', 'PGD Attack'],
        'Accuracy': [clean_acc, fgsm_acc, pgd_acc],
        'Inference Time': [clean_time, fgsm_time, pgd_time]
    })
    metrics_df.to_csv('results/metrics_comparison.csv', index=False)

def print_final_summary(clean_acc, fgsm_acc, pgd_acc, train_time, clean_time, fgsm_time, pgd_time):
    """Print final summary."""
    print("\n=== Final Summary ===")
    print("\nAccuracy Metrics:")
    print(f"Initial clean accuracy: {clean_acc:.4f}")
    print(f"Accuracy under FGSM attack: {fgsm_acc:.4f} (Drop: {(clean_acc - fgsm_acc):.4f})")
    print(f"Accuracy under PGD attack: {pgd_acc:.4f} (Drop: {(clean_acc - pgd_acc):.4f})")
    print("\nPerformance Metrics:")
    print(f"Standard training time: {train_time:.2f} seconds")
    print(f"Average clean inference time: {clean_time:.4f} seconds per batch")
    print(f"Average FGSM attack+inference time: {fgsm_time:.4f} seconds per batch")
    print(f"Average PGD attack+inference time: {pgd_time:.4f} seconds per batch")

def generate_final_report(clean_acc, fgsm_acc, pgd_acc):
    """Generate final report."""
    with open('results/txt/final_report.txt', 'w') as f:
        f.write("=== Adversarial Attack Benchmark Report ===\n\n")
        f.write("Key Findings:\n")
        f.write(f"- The model's accuracy drops from {clean_acc:.2%} to {fgsm_acc:.2%} under FGSM attack ({((clean_acc - fgsm_acc) / clean_acc):.2%} reduction)\n")
        f.write(f"- Under PGD attack, accuracy drops to {pgd_acc:.2%}\n")
        f.write(f"- The average inference time increases significantly under attacks, indicating the computational overhead of generating adversarial examples\n\n")

def visualize_attacks(model, test_loader, device, art_classifier, num_examples=5):
    """Visualize attacks."""
    model.eval()
    os.makedirs("img/attacks", exist_ok=True)
    inputs, labels = next(iter(test_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    fgsm = FastGradientMethod(art_classifier, eps=EPSILON)
    pgd = ProjectedGradientDescent(art_classifier, eps=EPSILON, max_iter=10)
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
        plt.title("FGSM Perturbed ε="+str(EPSILON))
        plt.axis('off')
        plt.subplot(num_examples, 5, i * 5 + 3)
        plt.imshow(fgsm_diff, cmap='hot')
        plt.title("FGSM Difference")
        plt.axis('off')
        plt.subplot(num_examples, 5, i * 5 + 4)
        plt.imshow(pgd_img, cmap='gray')
        plt.title("PGD Perturbed ε="+str(EPSILON))
        plt.axis('off')
        plt.subplot(num_examples, 5, i * 5 + 5)
        plt.imshow(pgd_diff, cmap='hot')
        plt.title("PGD Difference")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('img/attacks/attack_visualization.png')
    plt.close()
    for i, idx in enumerate(indices):
        fig, axs = plt.subplots(1, 5, figsize=(24, 4))
        original_img = inputs[idx].cpu().squeeze().numpy()
        fgsm_img = x_adv_fgsm[idx].cpu().squeeze().numpy()
        pgd_img = x_adv_pgd[idx].cpu().squeeze().numpy()
        fgsm_diff = np.abs(original_img - fgsm_img)
        pgd_diff = np.abs(original_img - pgd_img)
        axs[0].imshow(original_img, cmap='gray')
        axs[0].set_title(f"Original (Label: {labels[idx].item()})")
        axs[0].axis('off')
        axs[1].imshow(fgsm_img, cmap='gray')
        axs[1].set_title("FGSM Perturbed ε="+str(EPSILON))
        axs[1].axis('off')
        axs[2].imshow(fgsm_diff, cmap='hot')
        axs[2].set_title("FGSM Difference")
        axs[2].axis('off')
        axs[3].imshow(pgd_img, cmap='gray')
        axs[3].set_title("PGD Perturbed ε="+str(EPSILON))
        axs[3].axis('off')
        axs[4].imshow(pgd_diff, cmap='hot')
        axs[4].set_title("PGD Difference")
        axs[4].axis('off')
        plt.tight_layout()
        plt.savefig(f'img/attacks/attack_example_{i}.png')
        plt.close()

def main():
    setup_environment()
    create_csv_if_not_exists()
    shuffle_csv('adni_dataset2/train.csv')
    device = set_device()
    transform = transforms.Compose([
        transforms.Resize((200, 190)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    (train_loader, test_loader, train_dataset, test_dataset), prep_time = prepare_data('adni_dataset2/train.csv', 'adni_dataset2/AugmentedAlzheimerDataset', transform)
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("[*] Training initial model...")
    train_losses, train_time = train_model(model, train_loader, criterion, optimizer, device)
    plot_training_metrics(train_losses, train_time)
    (clean_acc, clean_cm, clean_time), eval_time = evaluate_model(model, test_loader, device, criterion)
    plot_confusion_matrix(clean_cm, clean_acc, 'Initial Confusion Matrix', 'img/initial_confusion_matrix.png')
    art_classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 160, 256),
        nb_classes=4,
        clip_values=(0, 1)
    )
    print("\n=== Evasion Attacks ===")
    (fgsm_acc, fgsm_cm, fgsm_time), fgsm_eval_time = test_evasion_attack(FastGradientMethod(art_classifier, eps=EPSILON), f"FGSM (ε={EPSILON})", test_loader, device, model)
    (pgd_acc, pgd_cm, pgd_time), pgd_eval_time = test_evasion_attack(ProjectedGradientDescent(art_classifier, eps=EPSILON, max_iter=10), f"PGD (ε={EPSILON}, iter=10)", test_loader, device, model)
    plot_attack_comparison(clean_cm, clean_acc, fgsm_cm, fgsm_acc, pgd_cm, pgd_acc)
    attack_metrics = {
        'Clean': {'accuracy': clean_acc, 'time': clean_time},
        'FGSM': {'accuracy': fgsm_acc, 'time': fgsm_time},
        'PGD': {'accuracy': pgd_acc, 'time': pgd_time}
    }
    plot_performance_metrics(attack_metrics)
    save_metrics(clean_acc, fgsm_acc, pgd_acc, clean_time, fgsm_time, pgd_time)
    print_final_summary(clean_acc, fgsm_acc, pgd_acc, train_time, clean_time, fgsm_time, pgd_time)
    generate_final_report(clean_acc, fgsm_acc, pgd_acc)
    print("\n[*] Generating attack visualizations...")
    visualize_attacks(model, test_loader, device, art_classifier)

if __name__ == "__main__":
    main()
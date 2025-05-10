import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import confusion_matrix

EPSILONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9]
MODEL_PATH = 'model/brain_mri_model.pth'

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
        img_path = os.path.join(self.root_dir, folder_name, img_name + '.jpg')
        image = Image.open(img_path).convert('L')
        y_label = torch.tensor(self.annotations.iloc[index, 1])
        if self.transform:
            image = self.transform(image)
        return image, y_label

def prepare_data(csv_file, root_dir, transform):
    dataset = BrainMRIDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset

class Net(nn.Module):
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

def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model, path=MODEL_PATH, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")
    return model

def calculate_metrics(original, perturbed):
    original_np = original.cpu().numpy().squeeze(1)
    perturbed_np = perturbed.cpu().numpy().squeeze(1)
    diff = np.abs(original_np - perturbed_np)
    percent_changed = np.mean(diff > 0) * 100
    ssim_value = ssim(original_np, perturbed_np, data_range=2.0)
    psnr_value = psnr(original_np, perturbed_np, data_range=2.0)
    return {
        'percent_changed': percent_changed,
        'ssim': ssim_value,
        'psnr': psnr_value
    }

def evaluate_accuracy(model, x_adv, y_true):
    model.eval()
    with torch.no_grad():
        outputs = model(x_adv)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_true).sum().item()
        return correct / len(y_true)

def visualize_mean_perturbation(inputs, perturbed, filename):
    mean_diff = torch.mean(torch.abs(perturbed - inputs), dim=0).squeeze().cpu().numpy()
    plt.imshow(mean_diff, cmap='hot')
    plt.title("Mean Perturbation")
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

def plot_distribution(values, title, filename):
    plt.figure()
    plt.hist(values, bins=50, color='purple', alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(filename)
    plt.close()

def plot_comparison_bars(data_dict, metric_name, save_path):
    plt.figure(figsize=(10, 6))
    attack_names = list(data_dict.keys())
    n = len(data_dict[attack_names[0]])

    bar_width = 0.35
    indices = np.arange(n)

    if isinstance(data_dict[attack_names[0]][0], dict):
        fgsm_vals = [m[metric_name] for m in data_dict['FGSM']]
        pgd_vals = [m[metric_name] for m in data_dict['PGD']]
    else:
        fgsm_vals = data_dict['FGSM']
        pgd_vals = data_dict['PGD']

    plt.bar(indices, fgsm_vals, width=bar_width, label='FGSM', alpha=0.7)
    plt.bar(indices + bar_width, pgd_vals, width=bar_width, label='PGD', alpha=0.7)

    plt.xlabel('Epsilon Index')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison by Attack')
    plt.xticks(indices + bar_width / 2, [str(eps) for eps in EPSILONS])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_attacks(model, test_loader, device, art_classifier, epsilons):
    model.eval()
    os.makedirs("img/advPatterns", exist_ok=True)
    os.makedirs("img/metrics", exist_ok=True)
    inputs, labels = next(iter(test_loader))
    inputs, labels = inputs.to(device), labels.to(device)

    attack_results, perturbation_norms, metrics, acc_results = {}, {'FGSM': [], 'PGD': []}, {'FGSM': [], 'PGD': []}, {'FGSM': [], 'PGD': []}

    for eps in epsilons:
        fgsm = FastGradientMethod(art_classifier, eps=eps)
        pgd = ProjectedGradientDescent(art_classifier, eps=eps, max_iter=10)
        x_adv_fgsm = torch.FloatTensor(fgsm.generate(inputs.cpu().numpy())).to(device)
        x_adv_pgd = torch.FloatTensor(pgd.generate(inputs.cpu().numpy())).to(device)

        attack_results[eps] = {'FGSM': x_adv_fgsm, 'PGD': x_adv_pgd}

        perturbation_norms['FGSM'].append(torch.norm((x_adv_fgsm - inputs).view(inputs.size(0), -1), dim=1).mean().item())
        perturbation_norms['PGD'].append(torch.norm((x_adv_pgd - inputs).view(inputs.size(0), -1), dim=1).mean().item())

        metrics['FGSM'].append(calculate_metrics(inputs, x_adv_fgsm))
        metrics['PGD'].append(calculate_metrics(inputs, x_adv_pgd))

        acc_results['FGSM'].append(evaluate_accuracy(model, x_adv_fgsm, labels))
        acc_results['PGD'].append(evaluate_accuracy(model, x_adv_pgd, labels))

        visualize_mean_perturbation(inputs, x_adv_fgsm, f"img/advPatterns/mean_perturbation_fgsm_eps{eps}.png")
        visualize_mean_perturbation(inputs, x_adv_pgd, f"img/advPatterns/mean_perturbation_pgd_eps{eps}.png")

        plot_distribution([m['ssim'] for m in metrics['FGSM']], f"FGSM SSIM Distribution ε={eps}", f"img/advPatterns/ssim_dist_fgsm_eps{eps}.png")
        plot_distribution([m['psnr'] for m in metrics['FGSM']], f"FGSM PSNR Distribution ε={eps}", f"img/advPatterns/psnr_dist_fgsm_eps{eps}.png")

    plot_comparison_bars(metrics, 'ssim', 'img/advPatterns/barplot_ssim_comparison.png')
    plot_comparison_bars(metrics, 'psnr', 'img/advPatterns/barplot_psnr_comparison.png')
    plot_comparison_bars(perturbation_norms, 'mean', 'img/advPatterns/barplot_perturbation_norms.png')
    plot_comparison_bars(acc_results, 'mean', 'img/advPatterns/barplot_accuracy_comparison.png')

def main():
    device = set_device()
    transform = transforms.Compose([
        transforms.Resize((200, 190)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_loader, test_loader, _, _ = prepare_data('adni_dataset2/train.csv', 'adni_dataset2/AugmentedAlzheimerDataset', transform)
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9)
    model = load_model(model, device=device)

    art_classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 200, 190),
        nb_classes=4,
        clip_values=(0, 1)
    )

    print("\n[*] Generating attack visualizations...")
    visualize_attacks(model, test_loader, device, art_classifier, EPSILONS)

if __name__ == "__main__":
    main()

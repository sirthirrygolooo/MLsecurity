import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
from art.defences.trainer import AdversarialTrainer
from art.defences.preprocessor import FeatureSqueezing, SpatialSmoothing
from torch.optim.lr_scheduler import ReduceLROnPlateau


class AdvancedAdversarialLab:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_environment()
        self.prepare_data()
        self.setup_model()
        self.attack_params = {
            'fgsm': {'eps': 0.2},
            'pgd': {'eps': 0.2, 'max_iter': 10},
            'carlini': {'confidence': 0.5, 'max_iter': 100}
        }
        self.setup_output_dirs()

    def setup_environment(self):
        torch.manual_seed(42)
        np.random.seed(42)
        print(f"[*] Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"[*] GPU: {torch.cuda.get_device_name(self.device)}")
            print(f"[*] CUDA: {torch.version.cuda}")
            print(f"[*] Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 3:.2f} GB")

    def setup_output_dirs(self):
        os.makedirs("img", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("models", exist_ok=True)

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = BrainMRIDataset(
            csv_file='adni_dataset/train.csv',
            root_dir='adni_dataset/ADNI_IMAGES/png_images',
            transform=transform
        )

        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

    def setup_model(self):
        self.model = EnhancedNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5)

    def train(self, epochs=15, adversarial=False):
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Adversarial training
                if adversarial:
                    inputs = self.generate_adversarial_batch(inputs, labels)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()

            # Validation
            val_loss, val_acc = self.validate()
            train_loss = epoch_train_loss / len(self.train_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "models/best_model.pth")

        return train_losses, val_losses

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                val_loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return val_loss / len(self.val_loader), correct / total

    def generate_adversarial_batch(self, inputs, labels):
        # Convert to ART format
        classifier = PyTorchClassifier(
            model=self.model,
            loss=self.criterion,
            optimizer=self.optimizer,
            input_shape=(1, 128, 128),
            nb_classes=5,
            clip_values=(0, 1)
        )

        # Randomly select attack type
        attack_type = np.random.choice(['fgsm', 'pgd'])

        if attack_type == 'fgsm':
            attack = FastGradientMethod(classifier, **self.attack_params['fgsm'])
        else:
            attack = ProjectedGradientDescent(classifier, **self.attack_params['pgd'])

        x_adv = attack.generate(inputs.cpu().numpy())
        return torch.FloatTensor(x_adv).to(self.device)

    def evaluate_defenses(self):
        # Load best model
        self.model.load_state_dict(torch.load("models/best_model.pth"))

        # Standard evaluation
        clean_acc, clean_cm = self.evaluate(self.test_loader)

        # Attack evaluations
        fgsm_acc, fgsm_cm = self.evaluate_attack('fgsm')
        pgd_acc, pgd_cm = self.evaluate_attack('pgd')
        cw_acc, cw_cm = self.evaluate_attack('carlini')

        # Defense evaluations
        defenses = {
            'feature_squeezing': FeatureSqueezing(),
            'spatial_smoothing': SpatialSmoothing()
        }

        defense_results = {}
        for name, defense in defenses.items():
            def_acc, def_cm = self.evaluate_with_defense(defense)
            defense_results[name] = (def_acc, def_cm)

        # Generate comprehensive report
        self.generate_report(
            clean_acc, clean_cm,
            {'fgsm': fgsm_acc, 'pgd': pgd_acc, 'carlini': cw_acc},
            {'fgsm': fgsm_cm, 'pgd': pgd_cm, 'carlini': cw_cm},
            defense_results
        )

    def evaluate(self, dataloader):
        self.model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        cm = confusion_matrix(all_labels, all_preds)
        return accuracy, cm

    def evaluate_attack(self, attack_type):
        classifier = PyTorchClassifier(
            model=self.model,
            loss=self.criterion,
            optimizer=self.optimizer,
            input_shape=(1, 128, 128),
            nb_classes=5,
            clip_values=(0, 1)
        )

        if attack_type == 'fgsm':
            attack = FastGradientMethod(classifier, **self.attack_params['fgsm'])
        elif attack_type == 'pgd':
            attack = ProjectedGradientDescent(classifier, **self.attack_params['pgd'])
        else:
            attack = CarliniL2Method(classifier, **self.attack_params['carlini'])

        self.model.eval()
        all_labels = []
        all_preds = []

        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            x_adv = attack.generate(inputs.cpu().numpy())
            adv_inputs = torch.FloatTensor(x_adv).to(self.device)

            with torch.no_grad():
                outputs = self.model(adv_inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        cm = confusion_matrix(all_labels, all_preds)
        return accuracy, cm

    def evaluate_with_defense(self, defense):
        classifier = PyTorchClassifier(
            model=self.model,
            loss=self.criterion,
            optimizer=self.optimizer,
            input_shape=(1, 128, 128),
            nb_classes=5,
            clip_values=(0, 1),
            preprocessing_defences=[defense]
        )

        self.model.eval()
        all_labels = []
        all_preds = []

        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Apply defense
            x_def, _ = defense(torch.unsqueeze(inputs, 0))
            x_def = torch.squeeze(x_def, 0)

            with torch.no_grad():
                outputs = self.model(x_def)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        cm = confusion_matrix(all_labels, all_preds)
        return accuracy, cm

    def generate_report(self, clean_acc, clean_cm, attack_accs, attack_cms, defense_results):
        # Visualization
        self.plot_results(clean_acc, clean_cm, attack_accs, attack_cms, defense_results)

        # Save metrics
        metrics = {
            'clean_accuracy': clean_acc,
            'attack_accuracies': attack_accs,
            'defense_accuracies': {k: v[0] for k, v in defense_results.items()}
        }

        with open("results/metrics.json", "w") as f:
            json.dump(metrics, f)

        # Generate text report
        with open("results/report.txt", "w") as f:
            f.write("=== Adversarial Robustness Report ===\n\n")
            f.write(f"Clean Accuracy: {clean_acc:.4f}\n\n")

            f.write("Attack Performance:\n")
            for name, acc in attack_accs.items():
                f.write(f"{name.upper()}: {acc:.4f} (Drop: {clean_acc - acc:.4f})\n")

            f.write("\nDefense Performance:\n")
            for name, (acc, _) in defense_results.items():
                f.write(f"{name}: {acc:.4f}\n")

            f.write("\nRecommendations:\n")
            f.write("- Implement adversarial training in production\n")
            f.write("- Combine feature squeezing with spatial smoothing\n")
            f.write("- Monitor for Carlini-Wagner attacks specifically\n")

    def plot_results(self, clean_acc, clean_cm, attack_accs, attack_cms, defense_results):
        # Accuracy comparison
        plt.figure(figsize=(12, 6))

        # Attack comparison
        plt.subplot(1, 2, 1)
        x = np.arange(len(attack_accs))
        plt.bar(x, [clean_acc] * len(x), width=0.3, label='Clean')
        plt.bar(x + 0.3, attack_accs.values(), width=0.3, label='Under Attack')
        plt.xticks(x + 0.15, attack_accs.keys())
        plt.title("Attack Impact on Accuracy")
        plt.legend()

        # Defense comparison
        plt.subplot(1, 2, 2)
        x = np.arange(len(defense_results))
        plt.bar(x, [clean_acc] * len(x), width=0.3, label='Clean')
        plt.bar(x + 0.3, [v[0] for v in defense_results.values()], width=0.3, label='With Defense')
        plt.xticks(x + 0.15, defense_results.keys())
        plt.title("Defense Effectiveness")
        plt.legend()

        plt.tight_layout()
        plt.savefig("img/results_comparison.png")


class EnhancedNet(nn.Module):
    def __init__(self):
        super(EnhancedNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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


if __name__ == "__main__":
    lab = AdvancedAdversarialLab()

    print("\n=== Standard Training ===")
    lab.train(epochs=15, adversarial=False)

    print("\n=== Adversarial Training ===")
    lab.train(epochs=15, adversarial=True)

    print("\n=== Evaluating Defenses ===")
    lab.evaluate_defenses()
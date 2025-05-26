import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
import shutil
import random

# Configuration
EPSILONS = [0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.4, 0.7, 0.9]
MODEL_PATH = 'model/brain_mri_model.pth'
INPUT_DIR = 'perturbed_dataset'
OUTPUT_DIR = 'test_dataset'
TEST_SIZE = 1000
NORMAL_RATIO = 0.8
POISONED_RATIO = 0.2

class TestDatasetCreator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((200, 190)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def load_model(self):
        model = Net().to(self.device)
        model = load_model(model, device=self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9)

        art_classifier = PyTorchClassifier(
            model=model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(1, 200, 190),
            nb_classes=4,
            clip_values=(0, 1)
        )
        return art_classifier

    def create_test_dataset(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for eps in EPSILONS:
            eps_dir = os.path.join(OUTPUT_DIR, f'eps_{eps}')
            os.makedirs(eps_dir, exist_ok=True)

            normal_dir = os.path.join(eps_dir, 'normal')
            poisoned_dir = os.path.join(eps_dir, 'poisoned')
            os.makedirs(normal_dir, exist_ok=True)
            os.makedirs(poisoned_dir, exist_ok=True)

            # Calculer le nombre d'images pour chaque catégorie
            num_normal = int(TEST_SIZE * NORMAL_RATIO)
            num_poisoned = TEST_SIZE - num_normal

            # Copier les images normales
            normal_src = os.path.join(INPUT_DIR, 'normal')
            normal_images = [f for f in os.listdir(normal_src) if f.endswith('.png')]
            selected_normal = random.sample(normal_images, min(num_normal, len(normal_images)))

            for img in selected_normal:
                src_path = os.path.join(normal_src, img)
                dst_path = os.path.join(normal_dir, img)
                shutil.copy(src_path, dst_path)

            poisoned_src = os.path.join(INPUT_DIR, f'eps_{eps}')
            poisoned_images = [f for f in os.listdir(poisoned_src) if f.endswith('.png')]
            selected_poisoned = random.sample(poisoned_images, min(num_poisoned, len(poisoned_images)))

            for img in selected_poisoned:
                src_path = os.path.join(poisoned_src, img)
                dst_path = os.path.join(poisoned_dir, img)
                shutil.copy(src_path, dst_path)

            print(f"Dataset created for epsilon {eps} with {len(selected_normal)} normal and {len(selected_poisoned)} poisoned images")

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

def load_model(model, path=MODEL_PATH, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")
    return model

if __name__ == "__main__":
    creator = TestDatasetCreator()
    creator.create_test_dataset()
    print(f"Test dataset created in {OUTPUT_DIR} directory")

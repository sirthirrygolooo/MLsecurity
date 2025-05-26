import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

EPSILONS = [0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.4, 0.7, 0.9]
MODEL_PATH = 'model/brain_mri_model.pth'
OUTPUT_DIR = 'perturbed_dataset'
NUM_IMAGES = 1000

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
    return dataset

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

def create_perturbed_dataset(model, dataset, device, art_classifier, epsilons):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'normal'), exist_ok=True)

    for eps in epsilons:
        os.makedirs(os.path.join(OUTPUT_DIR, f'eps_{eps}'), exist_ok=True)

    indices = np.random.choice(len(dataset), NUM_IMAGES, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)

    for i, (inputs, labels) in enumerate(loader):
        for j in range(inputs.size(0)):
            img = inputs[j].squeeze().cpu().numpy()
            img = (img * 0.5 + 0.5) * 255
            img = Image.fromarray(img.astype(np.uint8))
            img.save(os.path.join(OUTPUT_DIR, 'normal', f'normal_{i*32 + j}.png'))

    for eps in epsilons:
        print(f"Generating images for epsilon = {eps}")
        fgsm = FastGradientMethod(art_classifier, eps=eps)

        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)

            x_adv = torch.FloatTensor(fgsm.generate(inputs.cpu().numpy())).to(device)

            for j in range(x_adv.size(0)):
                img = x_adv[j].squeeze().cpu().numpy()
                img = (img * 0.5 + 0.5) * 255
                img = np.clip(img, 0, 255)
                img = Image.fromarray(img.astype(np.uint8))
                img.save(os.path.join(OUTPUT_DIR, f'eps_{eps}', f'eps_{eps}_{i*32 + j}.png'))

def main():
    device = set_device()
    transform = transforms.Compose([
        transforms.Resize((200, 190)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = prepare_data('adni_dataset2/train.csv', 'adni_dataset2/AugmentedAlzheimerDataset', transform)
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

    print("\n[*] Generating perturbed dataset...")
    create_perturbed_dataset(model, dataset, device, art_classifier, EPSILONS)
    print(f"Dataset generated in {OUTPUT_DIR} directory")

if __name__ == "__main__":
    main()

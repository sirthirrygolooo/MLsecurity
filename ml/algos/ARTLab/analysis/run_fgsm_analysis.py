import os
import random
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# === CONFIGURATION ===
original_dataset_path = '../adni_dataset2/AugmentedAlzheimerDataset'
output_root = 'final_stegano_fgsm_dataset'
os.makedirs(output_root, exist_ok=True)

num_images = 1000
percentage_attacked = 20  # % of images to attack
epsilon_values = [0.0, 0.01, 0.03, 0.05, 0.1]

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# === DATASET ===
dataset = datasets.ImageFolder(root=original_dataset_path, transform=transform)
indices = random.sample(range(len(dataset)), num_images)
subset = Subset(dataset, indices)
dataloader = DataLoader(subset, batch_size=1, shuffle=False)

# === DUMMY MODEL ===
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 128, 128)
            dummy_output = self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(dummy_input))))))
            flattened_size = dummy_output.view(-1).shape[0]
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

model = Net()
criterion = nn.CrossEntropyLoss()

# === FGSM ATTACK ===
def fgsm_attack(image, epsilon, label):
    image.requires_grad = True
    output = model(image)
    loss = criterion(output, torch.tensor([label]))
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    perturbed_image = image + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image.detach()

# === LSB STÉGANOGRAPHIE ===
def generate_random_message(length):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def insert_message_lsb(image_np, message):
    message += '$t3g0'
    message_bits = ''.join(format(ord(c), '08b') for c in message)
    message_bits += '0' * ((len(message_bits) % 8) % 8)

    flat = image_np.flatten()
    for i in range(min(len(message_bits), len(flat))):
        flat[i] = (flat[i] & 0xFE) | int(message_bits[i])
    return flat.reshape(image_np.shape)

# === TRAITEMENT PAR EPSILON ===
for epsilon in epsilon_values:
    print(f"\nTraitement pour epsilon = {epsilon}")
    epsilon_dir = os.path.join(output_root, f"epsilon_{epsilon}")
    img_dir = os.path.join(epsilon_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    excel_data = []
    attacked_indices = random.sample(range(num_images), int(num_images * percentage_attacked / 100))

    for i, (image, label) in tqdm(enumerate(dataloader), total=num_images, desc=f"Epsilon {epsilon}"):
        image = image.to(torch.float32)

        # Appliquer FGSM si nécessaire
        if i in attacked_indices and epsilon > 0:
            image_attacked = fgsm_attack(image.clone(), epsilon, label)
        else:
            image_attacked = image

        # Conversion pour LSB
        image_np = (image_attacked.squeeze().numpy() * 255).astype(np.uint8)
        message = generate_random_message(500)
        image_stego = insert_message_lsb(image_np.copy(), message)

        # Sauvegarde image
        image_pil = Image.fromarray(image_stego)
        image_path = os.path.join(img_dir, f"image_{i}.png")
        image_pil.save(image_path)

        # Ajout aux annotations
        excel_data.append({
            'id_code': i,
            'diagnosis': dataset.classes[label],
            'attacked': int(i in attacked_indices),
            'epsilon': epsilon,
            'message_size': len(message),
            'message': message
        })

    # Sauvegarde du CSV
    df = pd.DataFrame(excel_data)
    df.to_csv(os.path.join(epsilon_dir, 'train.csv'), index=False)

print("\n✅ Génération terminée pour toutes les valeurs d'epsilon.")

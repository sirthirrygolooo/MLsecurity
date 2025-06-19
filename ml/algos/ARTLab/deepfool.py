import os
import random
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np

original_dataset_path = 'adni_dataset2/AugmentedAlzheimerDataset'
steg_dataset_path = 'steg_dataset_df'
os.makedirs(steg_dataset_path, exist_ok=True)
excel_path = os.path.join(steg_dataset_path, 'train.csv')
img_path = os.path.join(steg_dataset_path, 'img')
os.makedirs(img_path, exist_ok=True)

percentage_altered = 20
size = 1000

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=original_dataset_path, transform=transform)
indices = random.sample(range(len(dataset)), size)
subset = torch.utils.data.Subset(dataset, indices)
dataloader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

num_images = len(subset)
num_altered = int(num_images * percentage_altered / 100)
altered_indices = random.sample(range(num_images), num_altered)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 128, 128)
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

model = Net()
criterion = nn.CrossEntropyLoss()

def deepfool_attack(image, model, num_classes=4, overshoot=0.02, max_iter=50):
    """
    Args:
        image: Input image tensor
        model: Model to attack
        num_classes: Number of classes in the model
        overshoot: Overshoot parameter for DeepFool
        max_iter: Maximum number of iterations

    """
    image = image.clone().detach()
    image.requires_grad = True

    fs = model(image)
    fs_list = [fs[0, i] for i in range(num_classes)]
    o = torch.argmax(fs.data, 1)

    iter = 0
    while o == torch.argmax(fs.data, 1) and iter < max_iter:
        pert = torch.zeros_like(image)
        fs[0, o].backward(retain_graph=True)
        grad_orig = image.grad.data.clone()

        for k in range(num_classes):
            if k == o:
                continue

            image.grad.data.zero_()
            fs[0, k].backward(retain_graph=True)
            grad = image.grad.data.clone()

            w = grad - grad_orig
            f = (fs[0, k] - fs[0, o]).data

            pert += (f.abs() / w.norm()) * w / w.norm()

        pert = (1 + overshoot) * pert / pert.norm()
        image.data += pert

        fs = model(image)
        o = torch.argmax(fs.data, 1)
        iter += 1

    return image.detach()

excel_data = []

for i, (image, label) in tqdm(enumerate(dataloader), total=num_images, desc="Traitement des images"):
    if i in altered_indices:
        perturbed_image = deepfool_attack(image, model)
        image_to_save = transforms.ToPILImage()(perturbed_image.squeeze())
        label_str = "atk"
    else:
        image_to_save = transforms.ToPILImage()(image.squeeze())
        label_str = "ras"

    image_path = os.path.join(img_path, f"image_{i}.png")
    image_to_save.save(image_path)

    excel_data.append({
        'id_code': i,
        'diagnosis': dataset.classes[label],
        'label': label_str
    })

df = pd.DataFrame(excel_data)
df.to_csv(excel_path, index=False)

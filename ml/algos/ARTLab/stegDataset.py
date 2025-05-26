import os
import random
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
from tqdm import tqdm

original_dataset_path = 'adni_dataset2/AugmentedAlzheimerDataset'
steg_dataset_path = 'steg_dataset'
os.makedirs(steg_dataset_path, exist_ok=True)
excel_path = os.path.join(steg_dataset_path, 'train.csv')
img_path = os.path.join(steg_dataset_path, 'img')
os.makedirs(img_path, exist_ok=True)

percentage_altered = 20
size = 10000

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

def pgd_attack(image, epsilon, alpha, num_iter, label):
    perturbed_image = image.clone().detach()
    for _ in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = criterion(output, torch.tensor([label]))
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image + alpha * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = perturbed_image.detach()
    return perturbed_image

epsilon = 0.1
alpha = 0.02
num_iter = 40

excel_data = []

for i, (image, label) in tqdm(enumerate(dataloader), total=num_images, desc="Traitement des images"):
    if i in altered_indices:
        perturbed_image = pgd_attack(image, epsilon, alpha, num_iter, label)
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

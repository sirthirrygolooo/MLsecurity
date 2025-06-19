import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths
original_dataset_path = 'adni_dataset2/AugmentedAlzheimerDataset'
steg_dataset_path = 'steg_dataset2'
img_path = os.path.join(steg_dataset_path, 'img')
excel_path = os.path.join(steg_dataset_path, 'train.csv')
output_dir = 'results/epsilon_analysis/'
model_dir = 'models/'

os.makedirs(img_path, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Parameters
dataset_size = 10000
batch_size = 32

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Dataset loading
dataset = datasets.ImageFolder(root=original_dataset_path, transform=transform)
indices = random.sample(range(len(dataset)), dataset_size)
subset = torch.utils.data.Subset(dataset, indices)
trainloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 128, 128)
            dummy_output = self.pool(torch.relu(self.conv2(self.pool(torch.relu(self.conv1(dummy_input))))))
            flat_size = dummy_output.view(-1).shape[0]
        self.fc1 = nn.Linear(flat_size, 512)
        self.fc2 = nn.Linear(512, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

# Training
def train_model(model, loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(10):
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

# FGSM Attack
def fgsm_attack(image, epsilon, label, model, criterion):
    if epsilon == 0:
        return image.clone()
    image.requires_grad = True
    output = model(image)
    loss = criterion(output, torch.tensor([label]))
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    perturbed = image + epsilon * data_grad.sign()
    return torch.clamp(perturbed, 0, 1)

# Train and load model
model = Net()
train_model(model, trainloader)
model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
model.eval()

# Experiment
epsilons = [0.0, 0.01, 0.03, 0.05]
message_sizes = [100, 200, 500, 1000, 2000, 5000]
criterion = nn.CrossEntropyLoss()

results = []

for epsilon in epsilons:
    for message_size in message_sizes:
        correct = 0
        total = 0
        start_time = time.time()

        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Eps={epsilon}, MsgSize={message_size}"):
            for j in range(len(images)):
                image = images[j].unsqueeze(0)
                label = labels[j]

                perturbed_image = fgsm_attack(image, epsilon, label, model, criterion)
                image_name = f"img_{i*len(images)+j}_eps_{epsilon}_msg_{message_size}.png"
                transforms.ToPILImage()(perturbed_image.squeeze()).save(os.path.join(img_path, image_name))

                output = model(perturbed_image)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == label).item()
                total += 1

                results.append({
                    'id_code': i*len(images)+j,
                    'diagnosis': dataset.classes[label],
                    'label': 'atk' if epsilon > 0 else 'clean',
                    'epsilon': epsilon,
                    'message_size': message_size,
                })

        accuracy = 100 * correct / total
        duration = time.time() - start_time

        results.append({
            'epsilon': epsilon,
            'message_size': message_size,
            'accuracy': accuracy,
            'execution_time_sec': duration,
        })
        print(f"Epsilon={epsilon}, Message Size={message_size}, Accuracy={accuracy:.2f}%, Time={duration:.2f}s")

# Save CSV
df = pd.DataFrame(results)
df.to_csv(excel_path, index=False)

# Plot Accuracy vs Epsilon
for message_size in message_sizes:
    plt.figure()
    subset = df[df['message_size'] == message_size].dropna()
    plt.plot(subset['epsilon'], subset['accuracy'], marker='o')
    plt.title(f'Accuracy vs Epsilon (Msg Size: {message_size})')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'accuracy_vs_epsilon_msg_{message_size}.png'))
    plt.close()

# Plot Accuracy vs Message Size
for epsilon in epsilons:
    plt.figure()
    subset = df[df['epsilon'] == epsilon].dropna()
    plt.plot(subset['message_size'], subset['accuracy'], marker='o')
    plt.title(f'Accuracy vs Message Size (Epsilon: {epsilon})')
    plt.xlabel('Message Size')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'accuracy_vs_msg_epsilon_{epsilon}.png'))
    plt.close()

# Plot Execution Time vs Epsilon
for message_size in message_sizes:
    plt.figure()
    subset = df[df['message_size'] == message_size].dropna()
    plt.plot(subset['epsilon'], subset['execution_time_sec'], marker='o', color='red')
    plt.title(f'Execution Time vs Epsilon (Msg Size: {message_size})')
    plt.xlabel('Epsilon')
    plt.ylabel('Time (sec)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'time_vs_epsilon_msg_{message_size}.png'))
    plt.close()

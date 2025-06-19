# train_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net
from torch.utils.data import DataLoader, Subset
import random

os.makedirs('models', exist_ok=True)
os.makedirs('data/augmented', exist_ok=True)


def train_for_message_size(message_size, data_path, model_save_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=os.path.join(data_path, f'msg_{message_size}'), transform=transform)
    indices = random.sample(range(len(dataset)), min(10000, len(dataset)))
    loader = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=True)

    model = Net()
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
        print(f"[MsgSize {message_size}] Epoch {epoch+1}/10 - Loss: {loss.item():.4f}")

    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_path, f'model.pth'))

if __name__ == '__main__':
    for message_size in [100, 200, 500, 1000, 2000, 5000]:
        train_for_message_size(message_size, 'data/augmented', f'models/msg_{message_size}')

# model.py
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 128)
            x = self.pool(torch.relu(self.conv1(dummy)))
            x = self.pool(torch.relu(self.conv2(x)))
            flat_size = x.view(-1).shape[0]

        self.fc1 = nn.Linear(flat_size, 512)
        self.fc2 = nn.Linear(512, 4)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(torch.relu(self.fc1(x)))
        return self.fc2(x)

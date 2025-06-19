import os
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True


original_dataset_path = 'email-phishing-dataset/email_phishing_data.csv'
steg_dataset_path = 'steg_phishing_dataset'
os.makedirs(steg_dataset_path, exist_ok=True)
excel_path = os.path.join(steg_dataset_path, 'train.csv')

# params
PERCENTAGE_ALTERED = 10
SAMPLE_SIZE = 10000
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
PGD_EPSILON = 0.02
PGD_ALPHA = 0.05
PGD_ITERS = 20

print("Chargement des données...")
data = pd.read_csv(original_dataset_path)

features = ['num_words', 'num_unique_words', 'num_stopwords', 'num_links',
            'num_unique_domains', 'num_email_addresses', 'num_spelling_errors',
            'num_urgent_keywords']
label = 'label'

X = data[features].values
y = data[label].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

train_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_train),
    torch.LongTensor(y_train)
)
test_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_test),
    torch.LongTensor(y_test)
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

class PhishingNet(nn.Module):
    def __init__(self, input_size):
        super(PhishingNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

input_size = X_train.shape[1]
model = PhishingNet(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Entraînement du modèle...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Époque {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

# PGD
def pgd_attack(model, sample, epsilon, alpha, num_iter, label):
    sample = sample.clone().detach().requires_grad_(True)
    original_sample = sample.clone().detach()

    for _ in range(num_iter):
        output = model(sample)

        loss = criterion(output, label.unsqueeze(0))

        loss.backward()

        perturbation = alpha * sample.grad.sign()

        perturbed_sample = sample + perturbation

        total_perturbation = perturbed_sample - original_sample
        total_perturbation = torch.clamp(total_perturbation, -epsilon, epsilon)
        sample = original_sample + total_perturbation

        sample = torch.clamp(sample, -3, 3)

        sample = sample.detach().requires_grad_(True)

    return sample.detach()

num_samples = min(SAMPLE_SIZE, len(test_dataset))
num_altered = int(num_samples * PERCENTAGE_ALTERED / 100)
altered_indices = random.sample(range(num_samples), num_altered)

print(f"Application des attaques PGD sur {num_altered} échantillons...")
excel_data = []
model.eval()

test_samples = [test_dataset[i] for i in range(num_samples)]

for i in tqdm(range(num_samples)):
    sample, label = test_samples[i]
    sample = sample.unsqueeze(0)  # [1, num_features]

    if i in altered_indices:
        perturbed_sample = pgd_attack(model, sample, PGD_EPSILON, PGD_ALPHA, PGD_ITERS, label)
        sample_np = perturbed_sample.squeeze().numpy()
        status = "atk"
    else:
        sample_np = sample.squeeze().numpy()
        status = "ras"

    original_values = scaler.inverse_transform(sample_np.reshape(1, -1))[0]
    original_values = np.round(original_values).astype(int)

    excel_data.append({
        'id': i,
        'label': label.item(),
        'status': status,
        **{features[j]: original_values[j] for j in range(len(features))}
    })

print("Sauvegarde des résultats...")
result_df = pd.DataFrame(excel_data)
result_df.to_csv(excel_path, index=False)
print(f"Résultats sauvegardés dans {excel_path}")

def evaluate_robustness(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

attacked_samples = []
attacked_labels = []

for data in excel_data:
    if data['status'] == 'atk':
        values = np.array([data[f] for f in features]).reshape(1, -1)
        scaled_values = scaler.transform(values)
        attacked_samples.append(scaled_values[0])
        attacked_labels.append(data['label'])

if attacked_samples:
    attack_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(np.array(attacked_samples)),
        torch.LongTensor(np.array(attacked_labels))
    )
    attack_loader = torch.utils.data.DataLoader(attack_dataset, batch_size=BATCH_SIZE)

    original_acc = evaluate_robustness(model, test_loader)
    attack_acc = evaluate_robustness(model, attack_loader)

    print("\nÉvaluation finale:")
    print(f"Précision sur les données originales: {original_acc:.4f}")
    print(f"Précision sur les données attaquées: {attack_acc:.4f}")
    print(f"Chute de performance: {(original_acc - attack_acc):.4f}")
else:
    print("\nAucun échantillon attaqué n'a été généré pour l'évaluation.")

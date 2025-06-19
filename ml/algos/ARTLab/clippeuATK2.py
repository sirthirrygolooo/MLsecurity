import os
import shutil
import pandas as pd
import torch
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm

# Configuration centralisée
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CLASSES = ["trafficlight", "stop", "speedlimit", "crosswalk"]
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    IMAGE_FOLDER = "RoadSign/images"
    ATK_IMAGE_FOLDER = "atk_roadsign"
    DATASET_ROOT = "dataset_combined"
    EXCEL_PATH = "RoadSign/predicted_classes.xlsx"

# Initialisation de la configuration
config = Config()

# Affichage du dispositif utilisé
print(f"[+] Utilisation de {torch.cuda.get_device_name(0) if config.DEVICE == 'cuda' else 'CPU'}")

# Transformation des images
normalize = transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

# Chargement du modèle CNN de base (ResNet18)
def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(config.CLASSES))
    model = model.to(config.DEVICE)
    model.eval()
    return model

model = load_model()

# Création des dossiers nécessaires
def create_directories():
    os.makedirs(os.path.join(config.DATASET_ROOT, "clean"), exist_ok=True)
    os.makedirs(os.path.join(config.DATASET_ROOT, "attacked"), exist_ok=True)

create_directories()

# Normalisation du nom de classe
def normalize_class_name(name):
    return name.lower().replace(" ", "").replace("sign", "")

# Génération de la map des labels à partir du fichier Excel
def generate_label_map_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    label_map = {}
    for _, row in df.iterrows():
        filename = row["Image"]
        pred_class_raw = row["Predicted Class"]
        normalized = normalize_class_name(pred_class_raw)
        if normalized in config.CLASSES:
            class_index = config.CLASSES.index(normalized)
            label_map[filename] = class_index
        else:
            print(f"[!] Classe inconnue ignorée : '{pred_class_raw}' (normalisée : '{normalized}')")
    return label_map

label_map = generate_label_map_from_excel(config.EXCEL_PATH)

# Recomposition du dataset
def recompose_dataset():
    print("[+] Recomposition du dataset avec les images originales et attaquées...")
    for image_file in tqdm(label_map):
        label_idx = label_map[image_file]
        label_name = config.CLASSES[label_idx]
        # Créer dossier pour la classe si nécessaire
        clean_class_dir = os.path.join(config.DATASET_ROOT, "clean", label_name)
        atk_class_dir = os.path.join(config.DATASET_ROOT, "attacked", label_name)
        os.makedirs(clean_class_dir, exist_ok=True)
        os.makedirs(atk_class_dir, exist_ok=True)
        src_clean = os.path.join(config.IMAGE_FOLDER, image_file)
        src_atk = os.path.join(config.ATK_IMAGE_FOLDER, f"atk_{image_file}")
        if os.path.exists(src_clean):
            shutil.copy(src_clean, os.path.join(clean_class_dir, image_file))
        if os.path.exists(src_atk):
            shutil.copy(src_atk, os.path.join(atk_class_dir, image_file))

# Évaluation du modèle
def evaluate(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def run_evaluation():
    print("[+] Évaluation sur les images originales et attaquées...")
    clean_dataset = ImageFolder(os.path.join(config.DATASET_ROOT, "clean"), transform=preprocess)
    atk_dataset = ImageFolder(os.path.join(config.DATASET_ROOT, "attacked"), transform=preprocess)
    clean_loader = DataLoader(clean_dataset, batch_size=32, shuffle=False)
    atk_loader = DataLoader(atk_dataset, batch_size=32, shuffle=False)
    acc_clean = evaluate(model, clean_loader)
    acc_atk = evaluate(model, atk_loader)
    print(f"\nAccuracy avant attaque : {acc_clean * 100:.2f}%")
    print(f"Accuracy après attaque : {acc_atk * 100:.2f}%")

if __name__ == "__main__":
    recompose_dataset()
    run_evaluation()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # Pour la barre de progression

model = ResNet50(weights='imagenet')
VERSION = "V2"

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    return np.argmax(predictions), np.max(predictions)

os.makedirs(f'results/classification_benchmark/{VERSION}', exist_ok=True)

epsilons = ['normal', 'eps_0.02', 'eps_0.03', 'eps_0.05', 'eps_0.2', 'eps_0.4', 'eps_0.7', 'eps_0.9']
results = []

# Remplacez ceci par vos vraies étiquettes de vérité terrain
ground_truth = np.random.randint(0, 1000, size=1000)

# Initialiser la barre de progression globale
total_images = len(epsilons) * 1000
progress_bar = tqdm(total=total_images, desc="Progression globale", unit="img")

for epsilon in epsilons:
    epsilon_dir = os.path.join('processed_dataset', epsilon)
    if not os.path.exists(epsilon_dir):
        print(f"Le dossier {epsilon_dir} n'existe pas, passage au suivant...")
        continue

    predictions = []
    confidences = []

    print(f"\nTraitement des images pour epsilon = {epsilon}...")

    epsilon_progress = tqdm(total=1000, desc=f"  Traitement {epsilon}", leave=False)

    for i in range(1000):
        img_path = os.path.join(epsilon_dir, f'{epsilon}_{i}.png')
        if not os.path.exists(img_path):
            epsilon_progress.update(1)
            progress_bar.update(1)
            continue

        pred_class, confidence = classify_image(img_path)
        predictions.append(pred_class)
        confidences.append(confidence)

        # Mettre à jour les barres de progression
        epsilon_progress.update(1)
        progress_bar.update(1)

    epsilon_progress.close()

    if predictions:  # Vérifier si des images ont été traitées
        accuracy = accuracy_score(ground_truth[:len(predictions)], predictions)
        avg_confidence = np.mean(confidences)
        results.append({
            'Epsilon': epsilon,
            'Accuracy': accuracy,
            'Confiance moyenne': avg_confidence,
            'Nombre d\'images': len(predictions)
        })

        print(f"Résultats pour {epsilon}: Accuracy = {accuracy:.4f}, Confiance moyenne = {avg_confidence:.4f}, Images traitées = {len(predictions)}")
    else:
        print(f"Aucune image traitée pour {epsilon}")

progress_bar.close()

# Sauvegarder les résultats dans un fichier CSV
results_df = pd.DataFrame(results)
results_df.to_csv(f'results/classification_benchmark/{VERSION}/results.csv', index=False)

print("\nRésultats de l'analyse :")
print(results_df)

# Visualisation 1: Accuracy en fonction de epsilon
plt.figure(figsize=(12, 6))
plt.plot(results_df['Epsilon'], results_df['Accuracy'], marker='o', linestyle='-', color='b')
plt.xlabel('Niveau de perturbation (epsilon)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Impact des perturbations FGSM sur l\'accuracy de classification', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0.9, color='r', linestyle='--', label='Seuil de dégradation modérée')
plt.axhline(y=0.7, color='orange', linestyle='--', label='Seuil de dégradation importante')
plt.legend()
plt.tight_layout()
plt.savefig(f'results/classification_benchmark/{VERSION}/accuracy_vs_epsilon.png')

# Visualisation 2: Confiance moyenne en fonction de epsilon
plt.figure(figsize=(12, 6))
plt.plot(results_df['Epsilon'], results_df['Confiance moyenne'], marker='o', linestyle='-', color='g')
plt.xlabel('Niveau de perturbation (epsilon)', fontsize=12)
plt.ylabel('Confiance moyenne', fontsize=12)
plt.title('Impact des perturbations FGSM sur la confiance du modèle', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'results/classification_benchmark/{VERSION}/confidence_vs_epsilon.png')

# Visualisation 3: Graphique combiné
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Niveau de perturbation (epsilon)', fontsize=12)
ax1.set_ylabel('Accuracy', color=color, fontsize=12)
ax1.plot(results_df['Epsilon'], results_df['Accuracy'], color=color, marker='o', linestyle='-')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Confiance moyenne', color=color, fontsize=12)
ax2.plot(results_df['Epsilon'], results_df['Confiance moyenne'], color=color, marker='o', linestyle='-')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Performance du modèle en fonction du niveau de perturbation', fontsize=14)
fig.tight_layout()
plt.savefig(f'results/classification_benchmark/{VERSION}/combined_metrics.png')


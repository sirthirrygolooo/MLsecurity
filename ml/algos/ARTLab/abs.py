import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
INPUT_DIR = 'perturbed_dataset'  # Dossier contenant votre dataset
OUTPUT_DIR = 'processed_dataset'  # Dossier de sortie
KERNEL_SIZE = 3  # Taille du voisinage pour le calcul de la moyenne (3x3)

def process_image(image_path, output_path):
    """Applique le traitement à une image et sauvegarde le résultat"""
    # Charger l'image en niveaux de gris
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32)

    # Créer une copie pour le résultat
    result = np.zeros_like(img_array)

    # Parcourir chaque pixel (en évitant les bords)
    for i in range(1, img_array.shape[0] - 1):
        for j in range(1, img_array.shape[1] - 1):
            # Extraire le voisinage 3x3
            neighborhood = img_array[i-1:i+2, j-1:j+2]

            # Calculer la moyenne des voisins (sans inclure le pixel central)
            neighbors = neighborhood.flatten()
            center_value = neighborhood[1, 1]
            neighbors_without_center = np.delete(neighbors, 4)  # Supprimer le pixel central
            mean_value = np.mean(neighbors_without_center)

            # Calculer la différence absolue
            result[i, j] = abs(center_value - mean_value)

    # Sauvegarder le résultat
    result_img = Image.fromarray(result.astype(np.uint8))
    result_img.save(output_path)

def process_dataset(input_dir, output_dir):
    """Traite toutes les images du dataset"""
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir tous les sous-dossiers
    for root, dirs, files in os.walk(input_dir):
        # Créer la structure de dossiers correspondante dans le dossier de sortie
        rel_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)

        # Traiter chaque image
        for file in tqdm(files, desc=f"Processing {rel_path}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subdir, file)
                process_image(input_path, output_path)

def visualize_example(input_path, output_path):
    """Visualise un exemple de traitement"""
    # Charger les images
    original = Image.open(input_path).convert('L')
    processed = Image.open(output_path).convert('L')

    # Afficher
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Image originale')
    plt.imshow(original, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Image traitée')
    plt.imshow(processed, cmap='gray')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Traiter le dataset
    process_dataset(INPUT_DIR, OUTPUT_DIR)

    # Visualiser un exemple
    example_input = os.path.join(INPUT_DIR, 'normal', os.listdir(os.path.join(INPUT_DIR, 'normal'))[0])
    example_output = os.path.join(OUTPUT_DIR, 'normal', os.listdir(os.path.join(INPUT_DIR, 'normal'))[0])
    visualize_example(example_input, example_output)

    print(f"Traitement terminé. Les images traitées sont dans {OUTPUT_DIR}")

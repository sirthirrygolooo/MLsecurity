import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

steg_dataset_path = 'steg_dataset'
noise_stegano_dataset_path = 'lsb_zero'
comparison_path = 'lsb_zero/comparison_images'
os.makedirs(comparison_path, exist_ok=True)
os.makedirs(noise_stegano_dataset_path, exist_ok=True)

excel_path = os.path.join(steg_dataset_path, 'train.csv')
img_path = os.path.join(steg_dataset_path, 'img')
new_excel_path = os.path.join(noise_stegano_dataset_path, 'train.csv')
new_img_path = os.path.join(noise_stegano_dataset_path, 'img')
os.makedirs(new_img_path, exist_ok=True)

annotations = pd.read_csv(excel_path)

def estimate_noise(image):
    high_pass_kernel = np.array([[-1, -1, -1],
                                 [-1,  8, -1],
                                 [-1, -1, -1]])
    high_pass = cv2.filter2D(image, -1, high_pass_kernel)
    noise_level = np.var(high_pass)
    return noise_level

def set_lsb_to_zero(image):
    return (image & 0xFE)

def visualize_comparison(original_image, modified_image, noise_before, noise_after, index):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Image originale')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(modified_image, cmap='gray')
    plt.title('Image avec LSB à 0')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    diff = cv2.absdiff(original_image, modified_image)
    plt.imshow(diff, cmap='gray')
    plt.title('Différence')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.hist(original_image.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title(f'Bruit avant (niveau: {noise_before:.2f})')
    plt.xlabel('Intensité')
    plt.ylabel('Fréquence')

    plt.subplot(2, 3, 5)
    plt.hist(modified_image.flatten(), bins=50, color='red', alpha=0.7)
    plt.title(f'Bruit après (niveau: {noise_after:.2f})')
    plt.xlabel('Intensité')
    plt.ylabel('Fréquence')

    plt.subplot(2, 3, 6)
    plt.hist(diff.flatten(), bins=50, color='green', alpha=0.7)
    plt.title('Différence de bruit')
    plt.xlabel('Intensité')
    plt.ylabel('Fréquence')

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_path, f'comparison_{index}.png'))
    plt.close()

new_data = []

for index, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Traitement des images"):
    img_file = os.path.join(img_path, f"image_{index}.png")
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    noise_level_before = estimate_noise(image)

    modified_image = set_lsb_to_zero(image)

    noise_level_after = estimate_noise(modified_image)

    #visualize_comparison(image, modified_image, noise_level_before, noise_level_after, index)

    new_img_file = os.path.join(new_img_path, f"image_{index}.png")
    cv2.imwrite(new_img_file, modified_image)

    new_data.append({
        'id_code': row['id_code'],
        'diagnosis': row['diagnosis'],
        'label': row['label'],
        'noise_level_before': noise_level_before,
        'noise_level_after': noise_level_after,
        'modification': 'LSB set to 0'
    })

new_annotations = pd.DataFrame(new_data)
new_annotations.to_csv(new_excel_path, index=False)

print("Nouveau dataset 'noise_stegano' créé avec succès.")

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import string
from tqdm import tqdm
from scipy.ndimage import generic_filter, median_filter

steg_dataset_path = 'steg_dataset'
noise_stegano_dataset_path = 'median_adaptive_stegano'
comparison_path = 'median_adaptive_stegano/comparison_images'
os.makedirs(comparison_path, exist_ok=True)
os.makedirs(noise_stegano_dataset_path, exist_ok=True)

excel_path = os.path.join(steg_dataset_path, 'train.csv')
img_path = os.path.join(steg_dataset_path, 'img')
new_excel_path = os.path.join(noise_stegano_dataset_path, 'train.csv')
new_img_path = os.path.join(noise_stegano_dataset_path, 'img')
os.makedirs(new_img_path, exist_ok=True)

annotations = pd.read_csv(excel_path)

def estimate_noise_median(image):
    median_img = median_filter(image, size=3)
    residual = image - median_img
    noise_level = np.var(residual)
    return noise_level

def calculate_texture_measure(image, x, y):
    neighborhood = image[max(0,x-1):x+2, max(0,y-1):y+2]
    return np.std(neighborhood)

def determine_message_size(noise_level):
    if noise_level < 10000:
        return 2000
    elif noise_level < 13000:
        return 500
    elif noise_level < 15000:
        return 100
    else:
        return 20

def generate_random_message(length):
    """Generate a random message of specified length"""
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))

def adaptive_steganography(image, message):
    # conversion message en bits
    message_bits = ''.join(format(ord(char), '08b') for char in message)
    message_index = 0
    stego_image = image.copy()
    height, width = image.shape

    # masque pour pas modif. plusieurs fois le même pixel
    used = np.zeros_like(image, dtype=bool)

    # boucle de balayage
    for x in range(1, height - 1):
        for y in range(1, width - 1):
            if message_index >= len(message_bits):
                return stego_image

            if used[x, y]:
                continue

            texture = calculate_texture_measure(image, x, y)

            # Seuil de texture : on encode seulement si la zone est texturée
            if texture > 10:
                original_pixel = image[x, y]
                lsb = int(message_bits[message_index])
                new_pixel = (original_pixel & ~1) | lsb  # Remplacer le LSB par le bit du message

                stego_image[x, y] = new_pixel
                used[x, y] = True
                message_index += 1

    # Si tout le message n'a pas été caché
    if message_index < len(message_bits):
        print(f"Attention : seulement {message_index} bits sur {len(message_bits)} ont été insérés.")

    return stego_image

def visualize_comparison(original_image, stego_image, noise_before, noise_after, index):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Image originale')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(stego_image, cmap='gray')
    plt.title('Image avec message adapté')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    diff = cv2.absdiff(original_image, stego_image)
    plt.imshow(diff, cmap='gray')
    plt.title('Différence')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.hist(original_image.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title(f'Bruit avant (niveau: {noise_before:.2f})')
    plt.xlabel('Intensité')
    plt.ylabel('Fréquence')

    plt.subplot(2, 3, 5)
    plt.hist(stego_image.flatten(), bins=50, color='red', alpha=0.7)
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

    noise_level_before = estimate_noise_median(image)
    message_size = determine_message_size(noise_level_before)

    message = generate_random_message(message_size)

    stego_image = adaptive_steganography(image, message)
    noise_level_after = estimate_noise_median(stego_image)

    new_img_file = os.path.join(new_img_path, f"image_{index}.png")
    cv2.imwrite(new_img_file, stego_image)

    new_data.append({
        'id_code': row['id_code'],
        'diagnosis': row['diagnosis'],
        'label': row['label'],
        'noise_level_before': noise_level_before,
        'noise_level_after': noise_level_after,
        'message_size': message_size,
        'message': message,
        'method': 'median_adaptive_steganography'
    })

new_annotations = pd.DataFrame(new_data)
new_annotations.to_csv(new_excel_path, index=False)

print("Nouveau dataset 'median_adaptive_stegano' créé avec succès.")

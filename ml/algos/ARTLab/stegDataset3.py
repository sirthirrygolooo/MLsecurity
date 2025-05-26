import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    if noise_level < 2000:
        return 2000
    elif noise_level < 2200:
        return 500
    elif noise_level < 2500:
        return 100
    else:
        return 20

def adaptive_steganography(image, message):
    message += '$t3g0'
    message_bits = ''.join(format(ord(char), '08b') for char in message)
    message_bits += '0' * ((len(message_bits) % 8) % 8)

    # Calculate texture map using median filter residual
    median_img = median_filter(image, size=3)
    residual = np.abs(image - median_img)
    texture_map = generic_filter(residual, np.std, size=3)

    # Normalize texture values for embedding strength
    texture_map = (texture_map - texture_map.min()) / (texture_map.max() - texture_map.min() + 1e-6)

    bit_index = 0
    stego_image = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if bit_index >= len(message_bits):
                break

            # Determine embedding strength based on texture (1-4 bits)
            embedding_strength = int(texture_map[i,j] * 3) + 1

            for k in range(min(embedding_strength, len(message_bits) - bit_index)):
                if bit_index >= len(message_bits):
                    break

                # Modify LSBs based on embedding strength
                mask = ~(1 << k)
                stego_image[i,j] = (stego_image[i,j] & mask) | (int(message_bits[bit_index]) << k)
                bit_index += 1

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
    message = "SecretMessage" * (message_size // 14)

    stego_image = adaptive_steganography(image, message)
    noise_level_after = estimate_noise_median(stego_image)

    # Visualization
    #visualize_comparison(image, stego_image, noise_level_before, noise_level_after, index)

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

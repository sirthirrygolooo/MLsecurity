import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

VERSION = '/P2/V2'

model = ResNet50(weights='imagenet')

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    predicted_class = decoded_predictions[0][1]
    confidence = decoded_predictions[0][2]
    return predicted_class, confidence

def denoise_image(img_path):
    img = image.load_img(img_path, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    denoised_img = denoise_tv_chambolle(img_array, weight=0.1)
    denoised_img = (denoised_img * 255).astype('uint8')
    return denoised_img

os.makedirs('results/denoising_analysis/%s' % VERSION, exist_ok=True)
os.makedirs('denoised_img/%s' % VERSION, exist_ok=True)

epsilons = ['normal', 'eps_0.02', 'eps_0.03', 'eps_0.05', 'eps_0.2', 'eps_0.4', 'eps_0.7', 'eps_0.9']
results = []

for epsilon in epsilons:
    epsilon_dir = os.path.join('processed_dataset', epsilon)
    if not os.path.exists(epsilon_dir):
        continue

    accuracies = []
    f1_scores = []
    true_classes = []
    predicted_classes_before = []
    predicted_classes_after = []

    for i in range(1000):
        img_path = os.path.join(epsilon_dir, f'{epsilon}_{i}.png')
        if not os.path.exists(img_path):
            continue

        predicted_class_before, confidence_before = classify_image(img_path)

        denoised_img = denoise_image(img_path)
        denoised_img_path = os.path.join(f'denoised_img/{VERSION}', f'denoised_{epsilon}_{i}.png')
        image.save_img(denoised_img_path, denoised_img)

        predicted_class_after, confidence_after = classify_image(denoised_img_path)

        true_class = predicted_class_before
        true_classes.append(true_class)
        predicted_classes_before.append(predicted_class_before)
        predicted_classes_after.append(predicted_class_after)

        accuracies.append((confidence_before, confidence_after))

    accuracy_before = accuracy_score(true_classes, predicted_classes_before)
    accuracy_after = accuracy_score(true_classes, predicted_classes_after)
    f1_before = f1_score(true_classes, predicted_classes_before, average='weighted')
    f1_after = f1_score(true_classes, predicted_classes_after, average='weighted')

    avg_confidence_before = np.mean([acc[0] for acc in accuracies])
    avg_confidence_after = np.mean([acc[1] for acc in accuracies])

    results.append((epsilon, avg_confidence_before, avg_confidence_after, accuracy_before, accuracy_after, f1_before, f1_after))

results_df = pd.DataFrame(results, columns=['Epsilon', 'Confiance avant débruitage', 'Confiance après débruitage', 'Accuracy avant débruitage', 'Accuracy après débruitage', 'F1-score avant débruitage', 'F1-score après débruitage'])
results_df.to_csv('results/denoising_analysis/%s/results.csv' % VERSION, index=False)

print("Résultats de l'analyse :")
print(results_df)

plt.figure(figsize=(10, 6))
plt.bar(results_df['Epsilon'], results_df['Accuracy avant débruitage'], width=0.4, label='Avant débruitage', align='center')
plt.bar([x + 0.4 for x in range(len(results_df['Epsilon']))], results_df['Accuracy après débruitage'], width=0.4, label='Après débruitage', align='center')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.title('Comparaison de l\'accuracy avant et après débruitage')
plt.xticks([x + 0.2 for x in range(len(results_df['Epsilon']))], results_df['Epsilon'])
plt.legend()
plt.savefig('results/denoising_analysis/%s/accuracy_comparison.png' % VERSION)

plt.figure(figsize=(10, 6))
plt.bar(results_df['Epsilon'], results_df['F1-score avant débruitage'], width=0.4, label='Avant débruitage', align='center')
plt.bar([x + 0.4 for x in range(len(results_df['Epsilon']))], results_df['F1-score après débruitage'], width=0.4, label='Après débruitage', align='center')
plt.xlabel('Epsilon')
plt.ylabel('F1-score')
plt.title('Comparaison du F1-score avant et après débruitage')
plt.xticks([x + 0.2 for x in range(len(results_df['Epsilon']))], results_df['Epsilon'])
plt.legend()
plt.savefig('results/denoising_analysis/%s/f1_score_comparison.png' % VERSION)

results_df['Gain de confiance'] = results_df['Confiance après débruitage'] - results_df['Confiance avant débruitage']
plt.figure(figsize=(10, 6))
plt.bar(results_df['Epsilon'], results_df['Gain de confiance'])
plt.xlabel('Epsilon')
plt.ylabel('Gain de confiance')
plt.title('Gain de confiance après débruitage')
plt.savefig('results/denoising_analysis/%s/gain.png' % VERSION)

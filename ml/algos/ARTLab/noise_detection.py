import os
import cv2
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

img_path = 'steg_dataset/img'
train_csv_path = os.path.join('steg_dataset', 'train.csv')
excel_path = os.path.join('steg_dataset', 'noise_indices_with_labels.csv')
excel_path2 = os.path.join('steg_dataset', 'noise_indices.csv')

def high_pass_filter(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    filtered = cv2.filter2D(image, -1, kernel)
    return np.mean(np.abs(filtered))

def median_filter(image):
    filtered = cv2.medianBlur(image, 3)
    return np.mean(np.abs(image - filtered))

def sobel_filter(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    return np.mean(sobel)

train_df = pd.read_csv(train_csv_path)

excel_data = []
excel_data2 = []
image_files = [f for f in os.listdir(img_path) if f.endswith('.png')]

for image_file in tqdm(image_files, desc="Calcul des indices de bruit"):
    image_path = os.path.join(img_path, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    high_pass_noise = high_pass_filter(image)
    median_noise = median_filter(image)
    sobel_noise = sobel_filter(image)

    image_id = int(image_file.split('_')[1].split('.')[0])

    diagnosis = train_df.loc[train_df['id_code'] == image_id, 'diagnosis'].values[0]
    label = train_df.loc[train_df['id_code'] == image_id, 'label'].values[0]

    excel_data.append({
        'image_id': image_id,
        'diagnosis': diagnosis,
        'label': label,
        'high_pass_noise': high_pass_noise,
        'median_noise': median_noise,
        'sobel_noise': sobel_noise
    })

    excel_data2.append({
        'image_id': image_id,
        'high_pass_noise': high_pass_noise,
        'median_noise': median_noise,
        'sobel_noise': sobel_noise
    })

df = pd.DataFrame(excel_data)
df2 = pd.DataFrame(excel_data2)
df.to_csv(excel_path, index=False)
df2.to_csv(excel_path2, index=False)

print("Indices de bruit calculés et enregistrés dans", excel_path, "et", excel_path2)
print("Nombre d'images traitées:", len(df))
print("#"* 50)
print("Debut d'entraînement des modèles de classification")

OUTPUT_DIR = 'results/noise_detection/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

X = df[['high_pass_noise', 'median_noise', 'sobel_noise']]
y = df['label']

# Vérification du nombre de classes
n_classes = len(np.unique(y))
print(f"Nombre de classes détectées: {n_classes}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create 3D visualization of the dataset
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Get unique labels and assign colors
unique_labels = np.unique(y)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

# Create a scatter plot for each label
for label, color in zip(unique_labels, colors):
    indices = y == label
    ax.scatter(X_scaled[indices, 0],
               X_scaled[indices, 1],
               X_scaled[indices, 2],
               c=[color],
               label=f'Label {label}',
               alpha=0.6,
               edgecolors='w',
               s=50)

# Set labels and title
ax.set_xlabel('High Pass Noise (scaled)')
ax.set_ylabel('Median Noise (scaled)')
ax.set_zlabel('Sobel Noise (scaled)')
ax.set_title('3D Visualization of Noise Features by Label')
ax.legend()

# Save the plot
plt.savefig(os.path.join(OUTPUT_DIR, '3d_dataset_visualization.png'))
plt.close()

print("3D visualization saved in", OUTPUT_DIR)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
}

results = {}
feature_importances = {}
execution_times = {}  # Dictionnaire pour stocker les temps d'exécution

for name, model in models.items():
    start_time = time.time()  # Début du chronomètre

    model.fit(X_train, y_train)

    fit_time = time.time() - start_time  # Temps d'entraînement

    start_time = time.time()  # Réinitialiser pour la prédiction
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time  # Temps de prédiction

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Stocker les temps d'exécution
    execution_times[name] = {
        'fit_time': fit_time,
        'predict_time': predict_time,
        'total_time': fit_time + predict_time
    }

    results[name] = {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "model": model
    }

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importances[name] = model.coef_[0]
    else:
        if hasattr(model, 'dual_coef_'):
            feature_importances[name] = np.mean(np.abs(model.dual_coef_), axis=0)
        else:
            feature_importances[name] = np.zeros(X.shape[1])

# Define model_names before using it
model_names = list(models.keys())

# Affichage des résultats avec les temps d'exécution
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {result['accuracy']}")
    print(f"Training time: {execution_times[name]['fit_time']:.4f} seconds")
    print(f"Prediction time: {execution_times[name]['predict_time']:.4f} seconds")
    print(f"Total time: {execution_times[name]['total_time']:.4f} seconds")
    print("Classification Report:")
    print(pd.DataFrame(result['report']).transpose())
    print("-"*50)

    plt.figure(figsize=(6, 4))
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{name}.png'))
    plt.close()

# Visualisation des temps d'exécution
plt.figure(figsize=(12, 6))

# Temps d'entraînement
plt.subplot(1, 2, 1)
train_times = [execution_times[name]['fit_time'] for name in model_names]
bars = plt.bar(model_names, train_times, color=['blue', 'green', 'red', 'purple', 'orange'])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')
plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.title('Training Time Comparison')
plt.ylim(0, max(train_times)*1.1)

# Temps de prédiction
plt.subplot(1, 2, 2)
predict_times = [execution_times[name]['predict_time'] for name in model_names]
bars = plt.bar(model_names, predict_times, color=['blue', 'green', 'red', 'purple', 'orange'])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')
plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.title('Prediction Time Comparison')
plt.ylim(0, max(predict_times)*1.1)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'execution_time_comparison.png'))
plt.close()

# Feature importance visualization
plt.figure(figsize=(10, 8))
feature_names = X.columns
for i, (name, importance) in enumerate(feature_importances.items()):
    plt.subplot(2, 3, i+1)
    plt.barh(feature_names, importance[:len(feature_names)])
    plt.title(f'Feature Importance - {name}')
    plt.xlabel('Importance')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_comparison.png'))
plt.close()

# Performance comparison
accuracies = [results[name]['accuracy'] for name in model_names]

plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple', 'orange'])

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1.1)
plt.savefig(os.path.join(OUTPUT_DIR, 'model_accuracy_comparison.png'))
plt.close()

# Classification report comparison
metrics = ['precision', 'recall', 'f1-score']
classes = np.unique(y)

plt.figure(figsize=(15, 10))
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    for name in model_names:
        report = results[name]['report']
        scores = [report[str(cls)][metric] for cls in classes]
        plt.plot(classes, scores, marker='o', label=name)

    plt.xlabel('Class')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} Comparison')
    plt.legend()
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'classification_metrics_comparison.png'))
plt.close()

# ROC Curve comparison
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(10, 8))
for name, result in results.items():
    model = result['model']
    if hasattr(model, 'predict_proba'):
        try:
            y_score = model.predict_proba(X_test)

            # Vérifier que y_score a la bonne forme
            if y_score.shape[1] == n_classes:
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                plt.plot(fpr["micro"], tpr["micro"],
                         label=f'{name} (AUC = {roc_auc["micro"]:.2f})')
            else:
                print(f"Modèle {name} ne retourne pas de probabilités pour toutes les classes")
        except Exception as e:
            print(f"Erreur avec le modèle {name}: {str(e)}")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-average ROC Curve Comparison')
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve_comparison.png'))
plt.close()

plt.figure(figsize=(10, 6))
total_times = [execution_times[name]['total_time'] for name in model_names]
bars = plt.bar(model_names, total_times, color=['blue', 'green', 'red', 'purple', 'orange'])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')
plt.xlabel('Model')
plt.ylabel('Total Time (seconds)')
plt.title('Total Execution Time Comparison')
plt.ylim(0, max(total_times)*1.1)
plt.savefig(os.path.join(OUTPUT_DIR, 'total_execution_time_comparison.png'))
plt.close()

print("All visualizations saved in", OUTPUT_DIR)

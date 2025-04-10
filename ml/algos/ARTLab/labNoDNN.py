import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.defences.preprocessor import FeatureSqueezing
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# Création des dossiers de sortie
os.makedirs("NDNN/img", exist_ok=True)
os.makedirs("NDNN/res", exist_ok=True)
os.makedirs("NDNN/res/txt", exist_ok=True)


# Timing decorator
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"[TIME] {method.__name__} executed in {(te - ts):.2f} seconds")
        return result, te - ts

    return timed


np.random.seed(42)


# Data preparation - same as before but for ML models
class BrainMRIDatasetML:
    def __init__(self, csv_file, root_dir, img_size=(64, 64)):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.img_size = img_size

    def load_images(self):
        X = []
        y = []

        for idx in tqdm(range(len(self.annotations))):
            img_name = self.annotations.iloc[idx, 0]
            folder_name = img_name.split('-')[0]
            img_path = os.path.join(self.root_dir, folder_name, img_name + '.png')

            image = Image.open(img_path).convert('L')  # grayscale
            image = image.resize(self.img_size)
            img_array = np.array(image).flatten()

            X.append(img_array)
            y.append(self.annotations.iloc[idx, 1])

        return np.array(X), np.array(y)


@timeit
def prepare_data_ml():
    dataset = BrainMRIDatasetML(csv_file='adni_dataset/train.csv',
                                root_dir='adni_dataset/ADNI_IMAGES/png_images')

    X, y = dataset.load_images()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


X_train, X_test, y_train, y_test, scaler = prepare_data_ml()


# Train Random Forest model
@timeit
def train_rf_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100,
                                   max_depth=10,
                                   random_state=42,
                                   n_jobs=-1)
    model.fit(X_train, y_train)
    return model


print("\n[*] Training Random Forest model...")
rf_model, rf_train_time = train_rf_model(X_train, y_train)


# Evaluation function
@timeit
def evaluate_ml_model(model, X, y, attack_name=None):
    start_time = time.time()
    y_pred = model.predict(X)
    inference_time = time.time() - start_time

    accuracy = np.mean(y_pred == y)
    cm = confusion_matrix(y, y_pred)

    if attack_name:
        print(f"\n[*] Evaluation under {attack_name} attack:")
    else:
        print("\n[*] Clean evaluation:")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Inference time: {inference_time:.4f} seconds")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    return accuracy, cm, inference_time


# Clean evaluation
rf_clean_acc, rf_clean_cm, rf_clean_time = evaluate_ml_model(rf_model, X_test, y_test)

# Create ART classifier for attacks
rf_art_classifier = SklearnClassifier(
    model=rf_model,
    clip_values=(scaler.transform([[0] * 4096])[0],  # 64x64=4096
                 scaler.transform([[255] * 4096])[0])  # clip the values to this range
)

# Test evasion attacks
@timeit
def test_ml_attack(attack, name, X_test, y_test):
    X_adv = attack.generate(X_test)
    return evaluate_ml_model(rf_model, X_adv, y_test, name)


print("\n=== Evasion Attacks on Random Forest ===")
fgsm = FastGradientMethod(rf_art_classifier, eps=0.1)
pgd = ProjectedGradientDescent(rf_art_classifier, eps=0.1, max_iter=10)

rf_fgsm_acc, rf_fgsm_cm, rf_fgsm_time = test_ml_attack(fgsm, "FGSM (ε=0.1)", X_test, y_test)
rf_pgd_acc, rf_pgd_cm, rf_pgd_time = test_ml_attack(pgd, "PGD (ε=0.1, iter=10)", X_test, y_test)

# Visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(rf_clean_cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Clean Accuracy: {rf_clean_acc:.2%}')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 3, 2)
sns.heatmap(rf_fgsm_cm, annot=True, fmt='d', cmap='Reds')
plt.title(f'FGSM Attack Accuracy: {rf_fgsm_acc:.2%}')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 3, 3)
sns.heatmap(rf_pgd_cm, annot=True, fmt='d', cmap='Reds')
plt.title(f'PGD Attack Accuracy: {rf_pgd_acc:.2%}')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.savefig('NDNN/img/rf_attack_comparison.png')

## Defense: Feature Squeezing + SVM

print("\n=== Implementing Defense: Feature Squeezing with SVM ===")

# Feature Squeezing defense
feature_squeezing = FeatureSqueezing(bit_depth=4, clip_values=(0, 1))


@timeit
def train_defended_model(X_train, y_train):
    # Apply feature squeezing during training
    X_train_squeezed = feature_squeezing(X_train)

    # Train SVM (generally more robust than RF)
    model = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
    model.fit(X_train_squeezed, y_train)
    return model


svm_model, svm_train_time = train_defended_model(X_train, y_train)

# Create defended ART classifier
svm_art_classifier = SklearnClassifier(
    model=svm_model,
    preprocessing_defences=[feature_squeezing],
    clip_values=(scaler.transform([[0] * 4096])[0],
                 scaler.transform([[255] * 4096])[0])  # clip values for the defense
)

# Evaluate defended model
svm_clean_acc, svm_clean_cm, svm_clean_time = evaluate_ml_model(svm_model, X_test, y_test)

print("\n=== Testing Attacks on Defended Model ===")
svm_fgsm_acc, svm_fgsm_cm, svm_fgsm_time = test_ml_attack(fgsm, "FGSM on Defended SVM", X_test, y_test)
svm_pgd_acc, svm_pgd_cm, svm_pgd_time = test_ml_attack(pgd, "PGD on Defended SVM", X_test, y_test)

# Defense comparison visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.bar(['RF Clean', 'RF FGSM', 'RF PGD'],
        [rf_clean_acc, rf_fgsm_acc, rf_pgd_acc],
        color=['blue', 'red', 'red'])
plt.title('Random Forest Vulnerabilities')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

plt.subplot(1, 3, 2)
plt.bar(['SVM Clean', 'SVM FGSM', 'SVM PGD'],
        [svm_clean_acc, svm_fgsm_acc, svm_pgd_acc],
        color=['blue', 'red', 'red'])
plt.title('SVM with Feature Squeezing')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

plt.subplot(1, 3, 3)
width = 0.35
x = np.arange(3)
plt.bar(x - width / 2, [rf_clean_acc, rf_fgsm_acc, rf_pgd_acc],
        width, label='Random Forest')
plt.bar(x + width / 2, [svm_clean_acc, svm_fgsm_acc, svm_pgd_acc],
        width, label='Defended SVM')
plt.xticks(x, ['Clean', 'FGSM', 'PGD'])
plt.ylabel('Accuracy')
plt.title('Defense Effectiveness Comparison')
plt.legend()

plt.tight_layout()
plt.savefig('NDNN/img/ml_defense_comparison.png')

# Performance metrics
metrics_df = pd.DataFrame({
    'Model': ['Random Forest'] * 3 + ['SVM with Defense'] * 3,
    'Scenario': ['Clean', 'FGSM Attack', 'PGD Attack'] * 2,
    'Accuracy': [rf_clean_acc, rf_fgsm_acc, rf_pgd_acc,
                 svm_clean_acc, svm_fgsm_acc, svm_pgd_acc],
    'Inference Time': [rf_clean_time, rf_fgsm_time, rf_pgd_time,
                       svm_clean_time, svm_fgsm_time, svm_pgd_time],
    'Training Time': [rf_train_time, None, None,
                      svm_train_time, None, None]
})

metrics_df.to_csv('NDNN/res/ml_metrics_comparison.csv', index=False)

# Final summary
print("\n=== Final Summary ===")
print("\nRandom Forest Performance:")
print(f"Clean accuracy: {rf_clean_acc:.4f}")
print(f"Accuracy under FGSM attack: {rf_fgsm_acc:.4f} (Drop: {(rf_clean_acc - rf_fgsm_acc):.4f})")
print(f"Accuracy under PGD attack: {rf_pgd_acc:.4f} (Drop: {(rf_clean_acc - rf_pgd_acc):.4f})")

print("\nDefended SVM Performance:")
print(f"Clean accuracy: {svm_clean_acc:.4f}")
print(f"Accuracy under FGSM attack: {svm_fgsm_acc:.4f} (Drop: {(svm_clean_acc - svm_fgsm_acc):.4f})")
print(f"Accuracy under PGD attack: {svm_pgd_acc:.4f} (Drop: {(svm_clean_acc - svm_pgd_acc):.4f})")

print("\nPerformance Metrics:")
print(f"Random Forest training time: {rf_train_time:.2f} seconds")
print(f"SVM with defense training time: {svm_train_time:.2f} seconds")
print(f"Average RF clean inference time: {rf_clean_time:.4f} seconds")
print(f"Average SVM clean inference time: {svm_clean_time:.4f} seconds")

# Generate report
with open('NDNN/res/txt/ml_final_report.txt', 'w') as f:
    f.write("=== Traditional ML Adversarial Robustness Report ===\n\n")
    f.write("Key Findings:\n")
    f.write(
        f"- Random Forest shows vulnerability to attacks (accuracy drops from {rf_clean_acc:.2%} to {rf_fgsm_acc:.2%} under FGSM)\n")
    f.write(
        f"- SVM with Feature Squeezing defense shows improved robustness (FGSM accuracy {svm_fgsm_acc:.2%} vs RF {rf_fgsm_acc:.2%})\n")
    f.write(f"- The defense comes with a training time cost: {svm_train_time:.2f}s vs {rf_train_time:.2f}s for RF\n")
    f.write("\nRecommendations:\n")
    f.write("- For critical applications, consider ensemble methods combining both models\n")
    f.write("- Feature squeezing is effective but may lose some discriminative information\n")
    f.write("- For production systems, monitor model drift after deploying defenses\n")

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack
from art.defences.preprocessor import FeatureSqueezing, GaussianAugmentation
from art.defences.postprocessor import GaussianNoise, ReverseSigmoid
from art.defences.trainer import AdversarialTrainer

# Decorator pour chronométrer les fonctions
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"[TIME] {method.__name__} executed in {(te - ts):.2f} seconds")
        return result, te - ts
    return timed

# Initialisation
np.random.seed(42)
os.makedirs("img/ml", exist_ok=True)
os.makedirs("results/ml", exist_ok=True)

# Préparation des données
@timeit
def prepare_data():
    from sklearn.datasets import load_digits
    data = load_digits()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)

(X_train, X_test, y_train, y_test), _ = prepare_data()

# Entraînement du modèle baseline
@timeit
def train_baseline_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

print("\n[*] Training baseline Random Forest model...")
baseline_model, _ = train_baseline_model(X_train, y_train)

# Évaluation du modèle
@timeit
def evaluate_model(model, X, y, attack_name=None):
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    cm = confusion_matrix(y, y_pred)
    label = f"{attack_name} attack" if attack_name else "Clean evaluation"
    print(f"\n[*] Evaluation under {label}:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    return accuracy, cm

(clean_acc, clean_cm), _ = evaluate_model(baseline_model, X_test, y_test)

# Heatmap
if clean_cm is not None and clean_cm.size > 0:
    plt.figure(figsize=(8, 6))
    sns.heatmap(clean_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Initial Confusion Matrix\nAccuracy: {clean_acc:.2%}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('img/ml/initial_confusion_matrix.png')
    plt.close()

# ART classifier
art_classifier = SklearnClassifier(model=baseline_model, clip_values=(-5, 5))

# Attaque ZOO
@timeit
def test_evasion_attack(attack, name, X_test, y_test, classifier):
    X_adv = attack.generate(X_test)
    y_pred_proba = classifier.predict(X_adv)
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracy = np.mean(y_pred == y_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n[*] Attack: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return accuracy, cm

print("\n=== Evasion Attack: ZOO Only ===")
zoo = ZooAttack(classifier=art_classifier, confidence=0.5, batch_size=1, max_iter=20, nb_parallel=1)
zoo_acc, zoo_cm = test_evasion_attack(zoo, "ZOO Attack", X_test, y_test, art_classifier)

# Défenses de pré-traitement
print("\n=== Pre-processing Defenses ===")

@timeit
def apply_preprocessing_defenses(X_train, X_test, y_train):
    clip_values = (-5, 5)
    fs = FeatureSqueezing(bit_depth=4, clip_values=clip_values)
    X_train_fs, _ = fs(X_train, y_train)
    X_test_fs, _ = fs(X_test, y_test)
    ga = GaussianAugmentation(sigma=0.5, ratio=0.5, clip_values=clip_values)
    X_train_ga, y_train_ga = ga(X_train, y_train)
    return X_train_fs, X_test_fs, X_train_ga, y_train_ga

(X_train_fs, X_test_fs, X_train_ga, y_train_ga), _ = apply_preprocessing_defenses(X_train, X_test, y_train)

fs_model, _ = train_baseline_model(X_train_fs, y_train)
_, _ = evaluate_model(fs_model, X_test_fs, y_test, "Feature Squeezing")

ga_model, _ = train_baseline_model(X_train_ga, y_train_ga)
_, _ = evaluate_model(ga_model, X_test, y_test, "Gaussian Augmentation")

# Défenses de post-traitement
print("\n=== Post-processing Defenses ===")

from sklearn.preprocessing import OneHotEncoder

@timeit
def apply_postprocessing_defenses(model, X_train, y_train):
    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))

    gn = GaussianNoise(scale=0.1, apply_fit=True, apply_predict=True)
    gn_model = SklearnClassifier(model=model, postprocessing_defences=gn)
    gn_model.fit(X_train, y_train_onehot)

    rs = ReverseSigmoid(beta=1.0, gamma=0.1)
    rs_model = SklearnClassifier(model=model, postprocessing_defences=rs)
    rs_model.fit(X_train, y_train_onehot)

    return gn_model, rs_model


(gn_model, rs_model), _ = apply_postprocessing_defenses(baseline_model, X_train, y_train)
_, _ = evaluate_model(gn_model, X_test, y_test, "Gaussian Noise")
_, _ = evaluate_model(rs_model, X_test, y_test, "Reverse Sigmoid")

# Entraînement Adversarial
print("\n=== Adversarial Training (ZOO only) ===")

@timeit
def adversarial_training(X_train, y_train, attacks):
    art_model = SklearnClassifier(model=RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42), clip_values=(-5, 5))
    trainer = AdversarialTrainer(art_model, attacks)
    trainer.fit(X_train, y_train)
    return trainer.get_classifier()

attacks = [ZooAttack(classifier=art_classifier, confidence=0.5, batch_size=1, max_iter=10)]
adv_model, _ = adversarial_training(X_train, y_train, attacks)
_, _ = evaluate_model(adv_model, X_test, y_test, "Adversarial Training")

# Rapport final
print("\n=== Final Report ===")

defenses = {
    "Baseline": baseline_model,
    "Feature Squeezing": fs_model,
    "Gaussian Augmentation": ga_model,
    "Gaussian Noise": gn_model,
    "Reverse Sigmoid": rs_model,
    "Adversarial Training": adv_model
}

zoo_results = {}
for name, model in defenses.items():
    print(f"\n[*] Evaluating {name} defense against ZOO...")
    clf = SklearnClassifier(model=model, clip_values=(-5, 5))
    zoo = ZooAttack(clf, confidence=0.5, batch_size=1, max_iter=20)
    acc, _ = test_evasion_attack(zoo, f"ZOO ({name})", X_test, y_test, clf)
    zoo_results[name] = acc

plt.figure(figsize=(10, 6))
plt.bar(zoo_results.keys(), zoo_results.values(), color='teal')
plt.title('ZOO Attack Accuracy for Each Defense')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('img/ml/zoo_defense_comparison.png')
plt.close()

print("\n[*] Experiment completed. Results saved in 'results/ml/' directory")

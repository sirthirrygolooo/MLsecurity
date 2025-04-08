#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical

# Chargement et préparation des données
digits = load_digits()
X = digits.images / 16.0  # Normalisation entre 0-1
y = digits.target
y_cat = to_categorical(y, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Modèle Keras avec Input layer explicite
def create_model():
    model = Sequential([
        Input(shape=(8, 8)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Création et entraînement du modèle
model = create_model()
print("Entraînement du modèle de base...")
model.fit(X_train, y_train,
          epochs=10,
          batch_size=32,
          validation_split=0.1,
          verbose=1)

# Évaluation initiale
_, clean_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAccuracy sur données propres: {clean_acc:.4f}")

# Fonction de visualisation
def plot_images(images, titles, shape=(8, 8)):
    plt.figure(figsize=(10, 3))
    for i in range(min(5, len(images))):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].reshape(shape), cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

############################## Attaques par évasion ##############################

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

# Encapsule le modèle Keras dans ART
art_classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)

# Attaque FGSM
print("\n=== Attaque FGSM ===")
attack_fgsm = FastGradientMethod(estimator=art_classifier, eps=0.2)
X_test_adv_fgsm = attack_fgsm.generate(X_test)

_, adv_acc_fgsm = model.evaluate(X_test_adv_fgsm, y_test, verbose=0)
print(f"Accuracy après FGSM: {adv_acc_fgsm:.4f} (drop: {clean_acc - adv_acc_fgsm:.4f})")

# Attaque PGD
print("\n=== Attaque PGD ===")
attack_pgd = ProjectedGradientDescent(estimator=art_classifier, eps=0.2, max_iter=10)
X_test_adv_pgd = attack_pgd.generate(X_test)

_, adv_acc_pgd = model.evaluate(X_test_adv_pgd, y_test, verbose=0)
print(f"Accuracy après PGD: {adv_acc_pgd:.4f} (drop: {clean_acc - adv_acc_pgd:.4f})")

# Visualisation
print("\nVisualisation des attaques:")
plot_images(X_test, ["Original"] * 5)
plot_images(X_test_adv_fgsm, ["FGSM"] * 5)
plot_images(X_test_adv_pgd, ["PGD"] * 5)

############################## Attaque par empoisonnement ##############################

from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd

print("\n=== Attaque par empoisonnement ===")
backdoor = PoisoningAttackBackdoor(perturbation=add_pattern_bd)

n_poison = int(0.1 * len(X_train))
poison_indices = np.random.choice(len(X_train), n_poison, replace=False)
X_poison = X_train[poison_indices].copy()
y_target = to_categorical(np.zeros(n_poison, dtype=int), num_classes=10)

X_poison_adv, y_poison_adv = backdoor.poison(X_poison, y_target)

X_train_poisoned = np.copy(X_train)
y_train_poisoned = np.copy(y_train)
X_train_poisoned[poison_indices] = X_poison_adv
y_train_poisoned[poison_indices] = y_poison_adv

# Entraînement modèle empoisonné
model_poisoned = create_model()
poisoned_classifier = KerasClassifier(model=model_poisoned, clip_values=(0, 1), use_logits=False)
model_poisoned.fit(X_train_poisoned, y_train_poisoned,
                   epochs=10,
                   batch_size=32,
                   validation_split=0.1,
                   verbose=1)

X_test_backdoor, _ = backdoor.poison(X_test[:5], to_categorical(np.zeros(5, dtype=int), num_classes=10))
predictions = np.argmax(model_poisoned.predict(X_test_backdoor), axis=1)
print("Prédictions sur images avec backdoor:", predictions)

# Visualisation
print("\nVisualisation de l'attaque par empoisonnement:")
plot_images(X_poison_adv[:5], ["Empoisonné"] * 5)
plot_images(X_test_backdoor, ["Backdoor"] * 5)

############################## Défenses ##############################

from art.defences.trainer import AdversarialTrainer

# Entraînement adversarial
print("\n=== Entraînement adversarial ===")
defender = AdversarialTrainer(classifier=art_classifier,
                              attacks=[attack_fgsm, attack_pgd],
                              ratio=0.5)
defender.fit(X_train, y_train, nb_epochs=15)

# Évaluation après défense
defended_model = defender.get_classifier().model
_, def_clean_acc = defended_model.evaluate(X_test, y_test, verbose=0)
_, def_adv_acc = defended_model.evaluate(X_test_adv_fgsm, y_test, verbose=0)
print(f"Après défense - Accuracy propre: {def_clean_acc:.4f}")
print(f"Après défense - Accuracy FGSM: {def_adv_acc:.4f}")

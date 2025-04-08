import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical

# Données
digits = load_digits()
X = digits.images / 16.0
y = digits.target
y_cat = to_categorical(y, num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Modèle
model = Sequential([
    Input(shape=(8, 8)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("👶 Entraînement du modèle de base...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Sauvegarde
model.save("model_digits.h5")
np.savez("digits_data.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

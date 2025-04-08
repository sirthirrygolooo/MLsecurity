import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

df = pd.read_csv('../../datasets/all_pokemon_data.csv')
df.fillna("None", inplace=True)

features = ['Primary Typing', 'Secondary Typing', 'Generation', 'Health', 'Attack',
            'Defense', 'Special Attack', 'Special Defense', 'Speed', 'Catch Rate',
            'Height (in)', 'Weight (lbs)']
target = 'Legendary Status'

X = df[features].copy()
y = df[target].copy()

for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
}

if xgb_available:
    models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

clean_results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    clean_results.append((name, acc))

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix for {name} (Clean Data)')
    plt.show()

X_train_corrupted = X_train.copy()
corrupted_indices = X_train_corrupted.sample(frac=0.1, random_state=42).index
X_train_corrupted.loc[corrupted_indices, 'Attack'] -= 30

corrupted_results = []
for name, model in models.items():
    model.fit(X_train_corrupted, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    corrupted_results.append((name, acc))

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix for {name} (Corrupted Data)')
    plt.show()

print("\nModel Performance on Clean Data:")
for name, acc in sorted(clean_results, key=lambda x: x[1], reverse=True):
    print(f"{name}: {acc:.2f}%")

print("\nModel Performance on Corrupted Data:")
for name, acc in sorted(corrupted_results, key=lambda x: x[1], reverse=True):
    print(f"{name}: {acc:.2f}%")

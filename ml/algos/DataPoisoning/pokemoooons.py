import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../../datasets/all_pokemon_data.csv')

# EDA
print(df.head())
print(df.info())
print(df.describe())

# Plotting
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Primary Typing')
plt.xticks(rotation=45)
plt.title('Distribution des Types Principaux de Pokémon')
plt.savefig('main_pokemons_type.png')

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Base Stat Total', bins=30, kde=True)
plt.title('Distribution du Total des Statistiques de Base')
plt.savefig('base_stat_distribution.png')

plt.figure(figsize=(10, 6))
correlation_matrix = df[['Health', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de Corrélation des Statistiques de Base')
plt.savefig('correlation_matrix.png')

df2 = df.copy()

label_encoder = LabelEncoder()
df2['Primary Typing'] = label_encoder.fit_transform(df2['Primary Typing'])
df2['Secondary Typing'] = label_encoder.fit_transform(df2['Secondary Typing'].astype(str))
df2['Generation'] = label_encoder.fit_transform(df2['Generation'])
df2['Legendary Status'] = label_encoder.fit_transform(df2['Legendary Status'].astype(str))
df2['Form'] = label_encoder.fit_transform(df2['Form'])
df2['Color ID'] = label_encoder.fit_transform(df2['Color ID'])

X = df2.drop(columns=['Name', 'Primary Typing'])
y = df2['Primary Typing']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Corrupt the training data
corrupted_indices = X_train.sample(frac=0.1, random_state=42).index
X_train_corrupted = X_train.copy()
X_train_corrupted.loc[corrupted_indices, 'Attack'] += 10

# Train models
model_clean = RandomForestClassifier(random_state=42)
model_clean.fit(X_train, y_train)

model_corrupted = RandomForestClassifier(random_state=42)
model_corrupted.fit(X_train_corrupted, y_train)

# Evaluate models
accuracy_clean = accuracy_score(y_test, model_clean.predict(X_test))
accuracy_corrupted = accuracy_score(y_test, model_corrupted.predict(X_test))

print(f"Accuracy with clean data: {accuracy_clean}")
print(f"Accuracy with corrupted data: {accuracy_corrupted}")

print("RAS")
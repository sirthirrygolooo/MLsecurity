# Explication des Attaques Adversaires dans ce Lab

Ce laboratoire se concentre sur les attaques adversaires par **évasion** (ou attaques au moment de l'inférence) contre un modèle de classification d'images IRM cérébrales. Voici une analyse détaillée des attaques et des défenses mises en œuvre:

## 1. Les Deux Types d'Attaques Implémentées

### a) FGSM (Fast Gradient Sign Method)
- **Principe**: Attaque basée sur le gradient, simple mais efficace
- **Fonctionnement**:
  - Calcule le gradient de la fonction de perte par rapport à l'image d'entrée
  - Ajoute une petite perturbation dans la direction qui maximise l'erreur
  - Formule: `x_adv = x + ε * sign(∇x J(θ, x, y))`
  - `ε` contrôle l'intensité de la perturbation (0.2 dans ce lab)

- **Caractéristiques**:
  - Attaque en une seule étape (one-shot)
  - Perturbations souvent visibles à l'œil nu
  - Rapide à calculer

### b) PGD (Projected Gradient Descent)
- **Principe**: Version itérative et plus puissante de FGSM
- **Fonctionnement**:
  - Applique FGSM en plusieurs petites étapes (10 itérations ici)
  - Après chaque étape, projette la perturbation dans une boule ε (pour limiter la magnitude)
  - Formule itérative: `x_adv(t+1) = Proj(x_adv(t) + α * sign(∇x J(θ, x_adv(t), y)))`
  
- **Caractéristiques**:
  - Attaque plus sophistiquée que FGSM
  - Génère des perturbations souvent plus subtiles
  - Plus coûteuse en calcul (10 itérations ici)

## 2. Impact des Attaques

Le code montre clairement l'impact via:
- La chute de précision après chaque attaque
- Les matrices de confusion qui révèlent comment les erreurs sont distribuées

**Exemple typique**:
- Précision initiale: 85%
- Après FGSM: Peut tomber à 30-40%
- Après PGD: Peut tomber encore plus bas (20-30%)

## 3. Défense: Adversarial Training

**Principe**:
- Entraîne le modèle sur un mélange d'exemples propres et adversaires
- Le modèle apprend à être robuste aux perturbations

**Implémentation dans le code**:
- Utilise `AdversarialTrainer` de ART
- Ratio de 0.5 (50% d'exemples adversaires pendant l'entraînement)
- Combine les deux attaques (FGSM et PGD) pour générer les exemples adversaires

**Effets observables**:
- La précision sur les exemples propres peut légèrement diminuer
- Mais la précision sur les attaques augmente significativement
- Le modèle devient plus robuste aux perturbations

## 4. Analyse des Résultats

Les matrices de confusion permettent de voir:
1. **Avant défense**:
   - Les attaques causent des erreurs systématiques
   - Certaines classes sont plus vulnérables que d'autres

2. **Après adversarial training**:
   - La diagonale principale est plus prononcée même sous attaque
   - Les erreurs sont plus aléatoires plutôt que systématiques

## 5. Implications Pratiques

- Les attaques adversaires peuvent tromper les modèles avec des perturbations imperceptibles
- Les modèles médicaux doivent être robustes à ces attaques pour des raisons de sécurité
- L'adversarial training est une défense efficace mais a un coût en performance

Ce lab illustre bien le "bras de fer" entre les attaquants qui développent des méthodes toujours plus sophistiquées et les défenseurs qui améliorent la robustesse des modèles.
*Sources : OWASP Top 10 Machine Learning ([lien](https://owasp.org/www-project-machine-learning-security-top-10/)), recherches complémentaires*

---

### **ML01:2023 Input Manipulation Attack**

**Description** : L’attaquant modifie les entrées envoyées au modèle pour tromper ses prédictions.  
**Sous-attaques** :

- **Adversarial Examples** : Ajout de perturbations imperceptibles.
    
- **Input Reconstruction Attacks** : Modifie l’entrée pour inverser un résultat.
    
- **Gradient-based Manipulation** : Utilise les gradients du modèle pour créer des entrées malicieuses.
    

---

### **ML02:2023 Data Poisoning Attack**

**Description** : Insertion de données malicieuses dans l’ensemble d’entraînement pour biaiser le modèle.  
**Sous-attaques** :

- **Availability Attacks** : Dégradent les performances globales.
    
- **Integrity Attacks** : Ciblent un comportement spécifique (ex. backdoors).
    
- **Clean-label Attacks** : Les données semblent normales mais affectent l’apprentissage.
    
- **Label Flipping** : Modifie les étiquettes de certaines données.
    

---

### **ML03:2023 Model Inversion Attack**

**Description** : L’attaquant déduit des informations sensibles sur les données d’entraînement à partir des sorties du modèle.  
**Sous-attaques** :

- **Gradient-based Inversion** : Utilise les gradients pour reconstruire des entrées.
    
- **Optimization-based Reconstruction** : Approxime les entrées par optimisation.
    
- **Attribute Inference** : Devine des attributs manquants d’une entrée partielle.
    

---

### **ML04:2023 Membership Inference Attack**

**Description** : L’attaquant détermine si une donnée donnée a été utilisée pour entraîner le modèle.  
**Sous-attaques** :

- **Black-box Attacks** : Utilisent uniquement les prédictions du modèle.
    
- **White-box Attacks** : Utilisent les paramètres internes du modèle.
    
- **Shadow Model Techniques** : Entraînent des modèles simulés pour inférer la présence.
    

---

### **ML05:2023 Model Theft**

**Description** : Extraction d’un modèle propriétaire en interrogeant ses prédictions.  
**Sous-attaques** :

- **Model Extraction via Querying** : Reproduit le comportement du modèle cible.
    
- **Knockoff Models** : Crée un clone approximatif.
    
- **API Exploitation** : Utilise l’accès API pour copier le modèle.
    

---

### **ML06:2023 AI Supply Chain Attacks**

**Description** : Compromettre un composant du pipeline IA (données, code, modèles).  
**Sous-attaques** :

- **Compromised Pre-trained Models** : Injecte des backdoors dans des modèles réutilisés.
    
- **Tainted Training Pipelines** : Corrompt les scripts ou environnements d'entraînement.
    
- **Dependency Hijacking** : Manipule les bibliothèques utilisées (ex. PyPI, GitHub).
    

---

### **ML07:2023 Transfer Learning Attack**

**Description** : Exploite les vulnérabilités introduites lors de la réutilisation de modèles pré-entraînés.  
**Sous-attaques** :

- **Backdoor Persistence** : Un backdoor dans le modèle source reste actif après fine-tuning.
    
- **Embedding Manipulation** : Modifie les couches de représentation pour biaiser le modèle.
    
- **Representation Inversion** : Récupère les données d'entraînement d'origine via les embeddings.
    

---

### **ML08:2023 Model Skewing**

**Description** : Introduction d’un biais subtil dans le modèle pour fausser certains résultats sans déclencher d’alerte.  
**Sous-attaques** :

- **Subtle Data Poisoning** : Affecte le modèle sans dégrader ses performances générales.
    
- **Targeted Misclassification** : Biaise le modèle sur un sous-ensemble spécifique.
    
- **Feedback Loop Exploitation** : Exploite des systèmes interactifs pour amplifier le biais.
    

---

### **ML09:2023 Output Integrity Attack**

**Description** : L’attaquant modifie ou intercepte les sorties du modèle.  
**Sous-attaques** :

- **Response Tampering** : Modifie la sortie avant qu’elle ne soit transmise à l’utilisateur.
    
- **Man-in-the-Middle (MitM)** : Intercepte les communications.
    
- **Result Injection** : Remplace ou insère des réponses malicieuses.
    

---

### **ML10:2023 Model Poisoning**

**Description** : Altère directement les poids ou la structure du modèle.  
**Sous-attaques** :

- **Weight Manipulation** : Modifie les poids du modèle pour altérer son comportement.
    
- **Backdoored Models** : Le modèle répond correctement en général mais se comporte mal sur des entrées spécifiques.
    
- **Trojan Insertion** : Insertion d’un déclencheur caché dans le modèle.
    

---

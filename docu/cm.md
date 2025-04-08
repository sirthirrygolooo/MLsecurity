Voici une représentation textuelle sous forme de carte mentale des différentes attaques contre les systèmes d'apprentissage automatique, leurs liens, forces et faiblesses, basée sur les sources fournies :

**Sécurité des Systèmes d'Apprentissage Automatique : Modèles d'Attaque**

*   **Data Poisoning (Empoisonnement des données)**
    *   **Fonctionnement :** Injection de données malveillantes dans l'entraînement. Altération du comportement du modèle. Données incorrectement étiquetées ou conçues pour induire des erreurs.
    *   **Objectifs :** Modifier le comportement du modèle. Réduire la précision globale. Induire des erreurs spécifiques. Manipuler pour des résultats biaisés. Classer erronément des exemples spécifiques (attaques ciblées). Ruiner la qualité du jeu de données (attaques aveugles).
    *   **Forces :** Très efficace avec accès aux données d'entraînement. Difficile à détecter si subtil.
    *   **Faiblesses :** Nécessite un accès aux données d'entraînement. Peut être détecté par validation croisée ou détection d'anomalies.
    *   **Types :**
        *   **Empoisonnement de données ciblées** : Classer erronément des exemples spécifiques pour des actions/omissions ciblées. Exemple : logiciel AV classé comme malveillant. Gravité : Critique.
        *   **Empoisonnement des données aveugles** : Ruiner la qualité/intégrité du jeu de données. Soucis avec les données publiques/non fiables. « Garbage in, garbage out ». Exemple : site web fiable compromis comme source de données. Gravité : Important.
    *   **Moyen :** Trojan compromettant/générant des données. Compromission de la source des données.

*   **Model Extraction & Inference attacks (Attaques d'extraction de modèle et d'inférence)**
    *   **Fonctionnement :** Interrogation du modèle pour le reconstruire ou extraire des informations. Analyse des réponses à de nombreuses requêtes.
    *   **Objectifs :** Voler la propriété intellectuelle (recréer le modèle). Comprendre les vulnérabilités. Accéder à des informations sensibles. Faciliter les attaques par évasion. Divulgation d'informations ciblées/secrètes.
    *   **Forces :** Réalisable sans accès direct au modèle (via requêtes). Permet de reproduire ou comprendre le modèle.
    *   **Faiblesses :** Nécessite beaucoup de requêtes (détection possible). Modèles complexes difficiles à extraire précisément.
    *   **Types :**
        *   **Model extraction (Extraction de modèle)** : Vol du modèle en interrogeant légitimement. Recréation du modèle sous-jacent.
            *   **Model inversion** : Récupération de fonctionnalités privées en optimisant la confiance retournée pour une cible. Aussi appelé « hill-climbing ».
            *   **Vol de modèle** : Fonctionnalité du nouveau modèle identique à l'original. Peut être inversé pour des informations/inférences.
            *   **Résolution d'équation** : Détermination de variables inconnues via les probabilités de classe de l'API.
            *   **Recherche de chemin** : Exploitation de l'API pour extraire les décisions d'un arbre de décision.
            *   **Attaque par transférabilité** : Apprentissage d'un modèle local pour créer des exemples contradictoires transférables au modèle cible.
        *   **Inference (Inférence)** : Extraction d'informations sensibles des prédictions.
            *   **Inférence d'abonnement** : Déterminer si un enregistrement faisait partie des données d'entraînement. Problème de confidentialité.

*   **Backdoor attacks (Attaques de porte dérobée)**
    *   **Fonctionnement :** Modification du modèle pour réagir spécifiquement à un déclencheur. Comportement malveillant activé uniquement par le déclencheur. Falsification des données d'entraînement par un tiers malveillant (MLaaS).
    *   **Objectifs :** Contrôle caché sur le modèle. Induire des erreurs spécifiques avec le déclencheur. Forcer des classifications incorrectes ciblées (ex: virus non malveillant).
    *   **Forces :** Difficile à détecter (comportement conditionnel). Très efficace pour les attaques ciblées.
    *   **Faiblesses :** Nécessite un accès à l'entraînement/déploiement. Peut être détecté par audits/tests rigoureux.
    *   **Origines :** Volontaire ou non (MLaaS, dépendances CVE/zero day). Compromission de dépendances tierces/mises à jour logicielles/autorités de certification.

*   **Adversarial attacks (Attaques adversariales)**
    *   **Fonctionnement :** Manipulation des entrées pour tromper le modèle. Création d'exemples adversariales (légèrement modifiés). Modifications souvent imperceptibles.
    *   **Objectifs :** Obtenir l'étiquette souhaitée pour une entrée. Induire des erreurs de classification/prédiction. Contourner les systèmes de sécurité basés sur le ML. Faux positifs/négatifs. Élévation des privilèges.
    *   **Forces :** Très efficace même avec des modifications mineures. Difficile à détecter (modifications subtiles). Peut ne pas laisser d'empreinte si les attaques d'apprentissage sont transférables.
    *   **Faiblesses :** Détection possible par des techniques spécifiques. Modifications doivent être soigneusement conçues. Impact négatif important sur l'exactitude.

**Liens entre les attaques :**

*   L'**extraction de modèle** peut précéder des **attaques adversariales** (fabrication hors ligne d'exemples contradictoires) ou des tentatives d'**inférence**.
*   Un modèle vulnérable à l'**empoisonnement de données** peut entraîner des faiblesses exploitables par d'autres types d'attaques.
*   Une **porte dérobée** peut être activée par une entrée spécialement conçue, partageant une certaine similitude avec les **attaques adversariales** (bien que l'objectif et la nature de la modification soient différents).

**Forces et Faiblesses Générales des Attaques :**

*   Les attaques nécessitant un accès direct aux données d'entraînement (comme l'**empoisonnement**) sont limitées par la sécurité de cet accès.
*   Les attaques basées sur l'interrogation (comme l'**extraction** et certaines **inférences**) peuvent être détectées par une surveillance du nombre de requêtes.
*   La subtilité des modifications est une force pour les **attaques adversariales** et l'**empoisonnement**, rendant la détection difficile.
*   La nécessité de concevoir des entrées spécifiques est une faiblesse pour les **attaques adversariales**.
*   Le déclencheur spécifique est à la fois une force (discrétion) et une faiblesse (nécessité d'introduire ce déclencheur) pour les **attaques de porte dérobée**.
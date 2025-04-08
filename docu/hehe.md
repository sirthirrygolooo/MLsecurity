**Synthèse des menaces et des mécanismes de défense pour la sécurisation des systèmes d'apprentissage automatique**

Les sources abordent la question cruciale de la sécurisation des systèmes d'apprentissage automatique, en analysant les différentes menaces potentielles et les mécanismes de défense associés.

**Modèles d'attaque**

Plusieurs types d'attaques ciblant les systèmes de ML sont détaillés dans les sources :

*   **Data Poisoning (Empoisonnement des données)** :
    *   **Fonctionnement :** Cette attaque consiste à introduire des données malveillantes ou altérées dans le jeu de données d'entraînement afin de modifier le comportement du modèle. Ces données peuvent être incorrectement étiquetées ou conçues pour induire des erreurs spécifiques. L'objectif est que le modèle généré lors de l'apprentissage fasse des prédictions erronées pendant la phase de test.
    *   **Objectifs :** Réduire la précision globale du modèle, induire des erreurs spécifiques sur certaines entrées, ou manipuler le modèle pour produire des résultats biaisés. Dans le cas d'attaques ciblées, l'attaquant souhaite spécifiquement que certains exemples soient mal classés pour forcer des actions ou des omissions spécifiques.
    *   **Types :** On distingue l'**empoisonnement de données ciblées** et l'**empoisonnement des données aveugles**. Dans le premier cas, l'attaquant vise des classifications erronées spécifiques, par exemple, en soumettant un logiciel antivirus comme malveillant pour le rendre inefficace. Dans le second cas, l'objectif est de ruiner la qualité ou l'intégrité du jeu de données, souvent lorsque celui-ci est public ou non traité, menant à une situation de type « garbage in, garbage out ». L'empoisonnement peut se faire via un trojan compromettant ou générant des données, ou indirectement en compromettant la source des données d'entraînement.
    *   **Gravité :** Critique pour l'empoisonnement ciblé et important pour l'empoisonnement aveugle.
    *   **Forces :** Peut être très efficace avec accès aux données d'entraînement et difficile à détecter si subtil.
    *   **Faiblesses :** Nécessite un accès aux données d'entraînement et peut être détecté par validation croisée ou détection d'anomalies.

*   **Model Extraction & Inference attacks (Attaques d'extraction de modèle et d'inférence)** :
    *   **Model extraction (Extraction de modèle)** : Vise à voler le modèle sous-jacent en le recréant par des requêtes légitimes. L'attaquant peut interroger le modèle avec des entrées choisies pour reconstruire le modèle ou en extraire des informations. Une fois extrait, le modèle peut être inversé pour récupérer des informations sur les caractéristiques ou opérer des inférences sur les données d'apprentissage. Des techniques comme la résolution d'équation via l'analyse des probabilités de classe retournées par une API, la recherche de chemin exploitant les particularités d'une API pour extraire les décisions d'un arbre de décision, et les attaques par transférabilité où un modèle local est appris pour fabriquer des exemples contradictoires transférables au modèle cible sont utilisées. L'extraction facilite également les attaques par évasion, notamment lorsque le modèle détecte des comportements hostiles. Un cas particulier est l'**inversion de modèle (model inversion)**, où des fonctionnalités privées utilisées dans les modèles peuvent être récupérées en recherchant l'entrée qui optimise le niveau de confiance renvoyé pour une classification cible.
    *   **Inference (Inférence)** : Consiste pour les attaquants à extraire des informations sensibles à partir des prédictions du modèle. L'**inférence d'abonnement** permet de déterminer si un enregistrement de données spécifique faisait partie du jeu de données d'apprentissage, soulevant des problèmes de confidentialité.
    *   **Objectifs :** Voler la propriété intellectuelle, comprendre les vulnérabilités du modèle, accéder à des informations sensibles utilisées par le modèle. L'extraction peut mener à la divulgation d'informations ciblées ou secrètes et faciliter la falsification non authentifiée des données du système.
    *   **Gravité :** Problème de confidentialité.
    *   **Forces :** Peut être réalisé sans accès direct au modèle et permet de reproduire le modèle ou d'en comprendre le comportement.
    *   **Faiblesses :** Nécessite un grand nombre de requêtes, ce qui peut être détecté, et les modèles complexes peuvent être difficiles à extraire avec précision.

*   **Backdoor attacks (Attaques de porte dérobée)** :
    *   **Fonctionnement :** Un attaquant modifie le modèle pour qu'il réagisse spécifiquement à un déclencheur particulier, activant un comportement malveillant uniquement lorsque ce déclencheur est présent. Cela peut être volontaire ou non, par exemple via un MLaaS ou des dépendances compromises. Le processus d'apprentissage peut être externalisé à un tiers malveillant qui falsifie les données et fournit un modèle avec un cheval de Troie forçant des classifications incorrectes ciblées.
    *   **Objectifs :** Permettre un contrôle caché sur le modèle et induire des erreurs spécifiques lorsque le déclencheur est présent. Les objectifs classiques incluent la compromission de dépendances tierces liées à la sécurité, la compromission de mécanismes de mise à jour logicielle ou d'autorités de certification.
    *   **Forces :** Difficile à détecter car le comportement malveillant est conditionnel et peut être très efficace pour des attaques ciblées.
    *   **Faiblesses :** Nécessite un accès au processus d'entraînement ou de déploiement et peut être détecté par des audits de sécurité ou des tests rigoureux.

*   **Adversarial attacks (Attaques adversariales)** :
    *   **Fonctionnement :** Elles consistent à manipuler les entrées pour tromper le modèle et obtenir des résultats erronés, en tentant d'obtenir l'étiquette souhaitée pour une entrée donnée. Les attaquants créent des entrées légèrement modifiées, appelées exemples adversariales, souvent imperceptibles pour les humains mais suffisantes pour tromper le modèle et provoquer des erreurs de classification ou de prédiction. Cela oblige généralement le modèle à renvoyer un faux positif ou un faux négatif, permettant une prise de contrôle subtile de la précision de la classification. Si l'attaque a un impact important sur l'exactitude, elle peut être plus longue à réaliser, nécessitant de manipuler et d'étiqueter spécifiquement les données sources.
    *   **Objectifs :** Induire des erreurs de classification ou de prédiction et contourner les systèmes de sécurité basés sur le ML. Un objectif courant est l'élévation des privilèges.
    *   **Gravité :** Critique.
    *   **Forces :** Peut être très efficace même avec des modifications mineures des entrées et difficile à détecter car les modifications sont subtiles. De plus, si le modèle est susceptible de transférer des attaques d'apprentissage forçant une classification incorrecte ciblée, il peut ne pas y avoir d'empreinte discernable du trafic de l'attaquant.
    *   **Faiblesses :** Peuvent être détectées par des techniques de défense spécifiques et les modifications doivent être soigneusement conçues pour être efficaces.

**Solutions**

Plusieurs solutions sont proposées pour renforcer la sécurité des systèmes de ML :

*   **Validation des données :** Techniques pour détecter et éliminer les données malveillantes avant l'entraînement.
*   **Robustesse du modèle :** Entraînement des modèles avec des techniques de régularisation et de généralisation pour les rendre plus résistants aux attaques.
*   **Surveillance continue :** Mise en place de systèmes de surveillance pour détecter les comportements anormaux en temps réel.
*   **Chiffrement des données :** Protection des données sensibles avec des techniques de chiffrement.

**Outils**

Des outils et plateformes existent pour aider à sécuriser les systèmes de ML :

*   **TensorFlow Extended (TFX) :** Plateforme de production ML avec fonctionnalités de sécurité.
*   **IBM Adversarial Robustness Toolbox (ART) :** Outils pour tester la robustesse des modèles.
*   **Microsoft Counterfit :** Bibliothèque pour évaluer et améliorer la sécurité des modèles de ML.
*   **OpenMined :** Plateforme open source pour le ML privé et sécurisé.

**Conclusion**

La sécurité des systèmes d'apprentissage automatique est un domaine complexe nécessitant une vigilance constante et l'adoption de techniques de défense adaptées pour protéger les modèles contre diverses menaces. La compréhension des différents types d'attaques est essentielle pour mettre en place des mesures de sécurité efficaces.
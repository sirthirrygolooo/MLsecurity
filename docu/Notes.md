# Notes sur sujet

**Sujet : **
>**Securing Machine Learning systems : Threat Analysis and Defense Mechanisms**

## Sujet complet

*A compléter*

## Modèles d'attaque 

- Data poisoning 
- Model extration & inference
- Backdoor ? Déterminer niveau et implication en ML
- Adversarial attacks 

## Data Poisoning

*Injection de données malveillantes dans le jeu de données d'entrainement, altération du comportement du modèle*

> Cas d'exemple : Modèle d'analyse de mails > Forçage du classement d'emails sans gravité en spam pour tromper le modèle

### Empoisonnement de données ciblées 

L'objectif de l'attaquant est de contaminer le modèle de machine généré lors de la phase d'apprentissage, de sorte que les prédictions sur de nouvelles données soient modifiées pendant la phase de test[1]. Dans des attaques par empoisonnement ciblées, l'attaquant souhaite classer erronément des exemples spécifiques pour faire en sorte que des mesures spécifiques soient prises ou omises.

**Exemple** : Soumission d'un logiciel AV en tant que logiciel malveillant pour forcer sa classification incorrecte comme malveillant et éliminer l'utilisation des logiciels AV ciblés sur les systèmes clients.

**Moyen** : Par le biais d'un trojan qui compromet, valide ou génère des données pour la création du modèle

**Gravité** : Critique

### Empoisonnent des données aveugles

L'objectif est de ruiner la qualité ou l'intégrité du jeu de données attaqué. De nombreux jeux de données sont publics, non fiables ou non traités. Cela suscite donc des inquiétudes supplémentaires quant à la capacité d'identifier les violations d'intégrité des données en premier lieu. Effectuer l'apprentissage sur des données compromises sans le savoir est une situation « garbage in, garbage out » (ordures à l'entrée, ordures à la sortie). Une fois la situation détectée, le triage doit déterminer l'étendue des données qui ont été violées ainsi que leur mise en quarantaine ou leur reformation.

**Exemple** : Une entreprise scrape un site web connu et fiable d'un expert sur un sujet donné pour en extraire des données d'entrainement. Le site web du fournisseur est compromis d'une manière ou d'une autre; l'attaquant peut donc empoisonner le jeu de données à sa guise sans s'attaquer directement au modèle.

**Gravité** : Important

## Model Extraction & Inference attacks

***Model extraction**: Vol de modèle pour pouvoir le *  
***Inference** : Extraction par les attaquants d'informations sensibles à partir des prédictions du modèle*

### Model extraction : model inversion

Les fonctionnalités privées utilisées dans des modèles d'apprentissage automatique peuvent être récupérées. Elles peuvent permettre, par exemple, de reconstituer des données d'apprentissage privées auxquelles l'attaquant n'a pas accès. Également connues sous le nom d'attaques de type « hill-climbing » dans la communauté biométrique, elles consistent à rechercher l'entrée qui optimise le niveau de confiance renvoyé, pour autant que la classification corresponde à la cible.

**Divulgation d'informations ciblées ou secrètes**

### Inférence d'abonnement 

L'attaquant peut déterminer si un enregistrement de données spécifique faisait ou non partie du jeu de données d'apprentissage du modèle. Les chercheurs ont pu prédire la procédure principale d'un patient (par exemple, l'opération chirurgicale qu'il a subie) en se basant sur les attributs (par exemple, l'âge, le sexe, l'hôpital).

**Point particulier**: Confidentialité des données. Des inférences sont effectuées à propos de l'inclusion d'un point de données dans le jeu d'apprentissage, mais les données d'apprentissage elles-mêmes ne sont pas divulguées.

**Gravité** : problème de confidentialité

### Vol de modèle

Les attaquants recréent le modèle sous-jacent en interrogeant le modèle de façon légitime. La fonctionnalité du nouveau modèle est identique à celle du modèle sous-jacent. Une fois le modèle recréé, il est possible de l'inverser afin de récupérer des informations sur des caractéristiques ou d'opérer des inférences sur des données d'apprentissage.

Résolution d'équation : pour un modèle qui retourne des probabilités de classe via une sortie d'API, un attaquant peut élaborer des requêtes visant à déterminer des variables inconnues dans un modèle.

Recherche de chemin : attaque consistant à exploiter les particularités d'une API pour extraire les « décisions » prises par un arbre de décision lors de la classification d'une entrée.

Attaque par transférabilité : un attaquant peut effectuer l'apprentissage d'un modèle local (par exemple, en adressant des requêtes de prédiction au modèle ciblé) et l'utiliser pour fabriquer des exemples contradictoires qui sont transférés vers le modèle cible. Si votre modèle est extrait et trouvé vulnérable à un type d'entrée contradictoire, de nouvelles attaques dirigées contre votre modèle déployé en production peuvent être développées entièrement hors connexion par l'attaquant qui a extrait une copie de votre modèle.

**Point particulier** : Quand un modèle d'apprentissage automatique sert à détecter des comportements hostiles, par exemple, pour identifier du courrier indésirable, classer des programmes malveillants et épingler des anomalies du réseau, une extraction du modèle peut faciliter des attaques par évasion

Falsification non authentifiée et en lecture seule des données du système, divulgation ciblée d'informations de grande valeur

## Backdoor attacks

***Backdoor**: Volontaire ou non, par le biais d'un MLaaS par exemple ou de par les dépendances (CVE ou zero day)*

Le processus d'apprentissage est externalisé à un tiers malveillant qui falsifie les données d'apprentissage et fournit un modèle doté d'un cheval de Troie qui force les classifications incorrectes ciblées, telles que la classification d'un virus donné comme non malveillant. Il s'agit d'un risque dans les scénarios de génération de modèle ML-as-a-service.

**Objectifs classiques** : Compromission d'une dépendance à l'égard d'un tiers liée à la sécurité / Mécanisme de mise à jour de logiciel compromis / Compromission de l'autorité de certification

## Adversarial attacks

***Evasion** : Manipulation des entrées pour tromper le modèle et obtenir des résultats erronés*

Cela se caractérise par la tentative d'un attaquant visant à obtenir d'un modèle qu'il lui renvoie l'étiquette souhaitée pour une entrée donnée. Cela oblige généralement un modèle à renvoyer un faux positif ou un faux négatif. Le résultat final est une prise de contrôle subtile de la précision de la classification du modèle, par laquelle un attaquant peut induire des contournements spécifiques à volonté.

Si cette attaque a un impact négatif important sur l'exactitude de la classification, elle peut aussi être plus longue à réaliser étant donné qu'un adversaire doit non seulement manipuler les données sources pour qu'elles ne soient plus étiquetées correctement, mais aussi les étiqueter spécifiquement avec l'étiquette frauduleuse souhaitée. Ces attaques impliquent souvent plusieurs étapes/tentatives pour forcer la classification incorrecte. Si le modèle est susceptible de transférer des attaques d'apprentissage qui forcent la classification incorrecte ciblée, il se peut qu'il n'y ait pas d'empreinte discernable du trafic de l'attaquant, car les attaques de détection peuvent être effectuées hors connexion.

**Objectif courant **: élévation des privilièges

**Gravité**: critique

### Dépendances d'exploit informatique

## Solutions

***Validation des données**: Techniques de validation pour détecter et éliminer les données malveillantes avant l'entrainement*  
***Robustesse du modèle**: Entrainement des modèles avec des techniques de régularisation et de généralisation pour les rendre + résistants aux attaques*  
***Surveillance continue**: Mise en place de systèmes de surveillance pour détecter les comportements anormaux en temps réel*  
***Chiffrement des données**: Protection des données sensibles avec des techniques de chiffrement*  

## Outils 

***TensorFlow Extended (TFX)**: Plateforme de production ML avec fonctionnalités de sécurité*  
***IBM Adversarial Robustness Toolbox (ART)**: Outils pour tester la robustess des modèles*  
***Microsoft Counterfit**: Une bibliothèque pour évaluer et améliorer la sécurité des modèles de ML*  
***OpenMined**: Plateforme opensource pour le ML privé et sécurisé*  

## Sources 
[Article présentation ML](https://www.livecampus.fr/blog-post/ia-et-machine-learning-dans-la-cybersecurite-comment-ils-faconneront-lavenir)  
[Article 1 securite machine learning](https://aventisec.com/blog/securite-machine-learning)  
[Article Microsoft](https://learn.microsoft.com/fr-fr/security/engineering/threat-modeling-aiml)  
[PDF 1](https://arxiv.org/pdf/2112.02797)
[PDF 2](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2025.pdf)

# Datasets 

https://www.kaggle.com/datasets/proutkarshtiwari/adni-images-for-alzheimer-detection  
https://www.kaggle.com/datasets/sarahtaha/1025-pokemon

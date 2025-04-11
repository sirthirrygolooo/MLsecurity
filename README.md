# Stage - Machine Learning et sécurité

## Présentation
Dataset : [ADNI alzheimer detection](https://www.kaggle.com/datasets/proutkarshtiwari/adni-images-for-alzheimer-detection) par [Utkarsh](https://www.kaggle.com/proutkarshtiwari)
## Analyses

## Résultats

### Attaques adversarial par inversion :

LAB en cours : [ici](./ml/algos/ARTLab)

Implémentation d'attaques **Fast Gradient Sign Method (FGSM)** et **Projected Gradient Descend (PGD)**

Caracérisation des images :
- 0: CN (Cognitively Normal)
- 1: MCI (Mild Cognitive Impairment)
- 2: EMCI (Early Mild Cognitive Impairment)
- 3: LMCI (Late Mild Cognitive Impairment)
- 4: AD (Alzheimer's Disease)

Premiers résultats :

Images après modif. par FGSM et PGD :  

**Exemple de FGSM sur images MNIST (ε=0.2)**  
![fgsm_attack.png](ml/algos/ARTLab/V1/img/fgsm_attack.png)

**FGSM (ε=0.2)**  
![fgsm_attack_full.png](ml/algos/ARTLab/V1/results/img/4V2/fgsm_attack_full.png)

**Exemple de PGD sur images MNIST (ε=0.2) 10 ité.**  
![pgd_attack.png](ml/algos/ARTLab/V1/img/pgd_attack.png)

**PGD (ε=0.2) 10 ité.**  
![pgd_attack_full.png](ml/algos/ARTLab/V1/results/img/4V2/pgd_attack_full.png)

**Matrices de confusion - Clean puis post-attaques**  
![attack_comparison.png](ml/algos/ARTLab/V1/results/img/4V2/attack_comparison.png)
![initial_confusion_matrix.png](ml/algos/ARTLab/V1/results/img/4V2/initial_confusion_matrix.png)
![fgsm_confusion_matrix.png](ml/algos/ARTLab/V1/results/img/4V2/fgsm_confusion_matrix.png)
![pgd_confusion_matrix.png](ml/algos/ARTLab/V1/results/img/4V2/pgd_confusion_matrix.png)

**MEP défense : adversarial training**

**Temps d'exécution comparaison**  
![training_time_comparison.png](ml/algos/ARTLab/V1/results/img/4V2/training_time_comparison.png)

**Efficacité de la défense**  
![defense_comparison.png](ml/algos/ARTLab/V1/results/img/4V2/defense_comparison.png)

**Matrices de confusion après défense**  

**FGSM**  
![defense_fgsm_confusion_matrix.png](ml/algos/ARTLab/V1/results/img/4V2/defense_fgsm_confusion_matrix.png)
**PGD**  
![defense_pgd_confusion_matrix.png](ml/algos/ARTLab/V1/results/img/4V2/defense_pgd_confusion_matrix.png)
## Bibliographie
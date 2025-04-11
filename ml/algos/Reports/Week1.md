<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Weekly Internship Report</title>
<style>
body {
  font-family: Arial, sans-serif;
  margin: 20px;
  line-height: 1.6;
  color: #333;
}
h1, h2, h3 {
  color: #444;
}
h1 {
  text-align: center;
  border-bottom: 2px solid #555;
  padding-bottom: 10px;
}
.section {
  margin-bottom: 30px;
}
.section-title {
  border-left: 5px solid #605f5f;
  padding-left: 10px;
  margin-bottom: 10px;
}
.section-content {
  padding: 10px;
  border-left: 2px solid #ddd;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
}
th, td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left;
}
th {
  background-color: #f9f9f9;
}
.img-container {
  margin: 15px 0;
  border: 1px solid #ccc;
  padding: 10px;
  background-color: #fafafa;
  text-align: center;
}
.img-container img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
}
.signature {
  text-align: right;
  margin-top: 20px;
  font-style: italic;
}
</style>
</head>
<body>

<h1>Weekly Internship Report</h1>

<div class="section">
<div class="section-title">
  <h2>General Information</h2>
</div>
<div class="section-content">
  <p><strong>Name:</strong> FROEHLY Jean-Baptiste</p>
  <p><strong>Period:</strong> Week 1 - 07/04/2025 to 11/04/2025</p>
</div>
</div>

<div class="section">
<div class="section-title">
  <h2>Weekly Objectives</h2>
</div>
<div class="section-content">
  <ul>
    <li><strong>Objective 1:</strong> Understand the subject and its challenges</li>
    <li><strong>Objective 2:</strong> Appropriate it and test initial things</li>
    <li><strong>Objective 3:</strong> Selection of an interesting and complete dataset</li>
    <li><strong>Objective 4:</strong> Conduct initial tests</li>
  </ul>
</div>
</div>

<div class="section">
<div class="section-title">
  <h2>Activities Completed</h2>
</div>
<div class="section-content">
  <h3>Monday</h3>
  <ul>
    <li>Research on different types of attacks, their mechanisms, and implications</li>
    <li>Analysis of each to see at which level of the ML chain they operate, their strengths and weaknesses</li>
    <li>Initial Data Poisoning tests on a small dataset (~1200 rows): not very conclusive</li>
  </ul>
  <h3>Tuesday</h3>
  <ul>
    <li>More in-depth research with a particular focus on Adversarial attacks</li>
    <li>Selection of more interesting datasets for my tests</li>
    <li>Development of a first lab based on an MRI image dataset for Alzheimer's disease detection</li>
  </ul>
  <h3>Wednesday</h3>
  <ul>
    <li>Improvement of the lab to implement initial attack cases with ART</li>
    <li>Focus on adversarial attacks by inversion</li>
  </ul>
  <h3>Thursday</h3>
  <ul>
    <li>Implementation of first adversarial attacks (NDNN and DNN), analysis of results</li>
    <li>Analyse du fonctionnement des attaques FGSM et PGD</li>
  </ul>
  <h3>Friday</h3>
  <ul>
    <li>Elaboration du rapport</li>
    <li>Defense mechanisms analysis on adversarial attacks through NDNN and DNN</li>
  </ul>
</div>
</div>
<div class="section">
<div class="section-title">
  <h2>Résumé</h2>
</div>
<div class="section-content">
  <p>Je me suis principalement concentré dans un premier temps sur les attaques <strong>Fast Gradient Sign Method</strong> et <strong>Projected Gradient Descent</strong></p>
  <h3>Fast Gradient Sign Method (FGSM)</h3>
  <ul>
    <li><strong>Principe:</strong> Attaque basée sur le gradient, simple mais efficace</li>
    <li><strong>Fonctionnement:</strong>
      <ul>
        <li>Calcule le gradient de la fonction de perte par rapport à l'image d'entrée</li>
        <li>Ajoute une petite perturbation dans la direction qui maximise l'erreur</li>
        <li>Formule: <code>x_adv = x + ε * sign(∇x J(θ, x, y))</code></li>
        <li>Intensité de la perturbation contrôlée par <code>ε</code> (<code>ε = 0.2</code> dans la plupart de mes tests)</li>
      </ul>
    </li>
    <li><strong>Caractéristiques:</strong>
      <ul>
        <li>Attaque en une seule étape (one-shot)</li>
        <li>Perturbations souvent visibles à l'œil nu</li>
        <li>Rapide à calculer</li>
      </ul>
    </li>
  </ul>
  <p>Exemple:</p>
  <div class="img-container">
    <img src="https://www.tensorflow.org/tutorials/generative/images/adversarial_example.png" alt="FGSM_illustration">
    <p>Illustration du principe FGSM (source TensorFlow.org) pour ε = .007</p>
  </div>
  <h3>Projected Gradient Descent (PGD)</h3>
  <ul>
    <li><strong>Principe:</strong> Version itérative plus puissante de FGSM</li>
    <li><strong>Fonctionnement:</strong>
      <ul>
        <li>Applique <strong>FGSM</strong> en plusieurs petites étapes (10 itérations dans mes tests)</li>
        <li>Après chaque étape, projette la perturbation dans un champ ε (pour limiter la magnitude et contrôler la perturbation)</li>
        <li>Formule itérative: <code>x_adv(t+1) = Proj(x_adv(t) + α * sign(∇x J(θ, x_adv(t), y)))</code></li>
      </ul>
    </li>
    <li><strong>Caractéristiques:</strong>
      <ul>
        <li>Plus sophistiquée que <strong>FGSM</strong></li>
        <li>Perturbations générées plus subtiles</li>
        <li>Plus coûteuse en calcul (10 it. dans mes tests)</li>
      </ul>
    </li>
  </ul>
  <table>
    <thead>
      <tr>
        <th>Experiment/Test</th>
        <th>Results</th>
        <th>Observations</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>[Name of experiment/test]</td>
        <td>[Results obtained]</td>
        <td>[Observations]</td>
      </tr>
    </tbody>
  </table>
</div>
</div>
<div class="section">
<div class="section-title">
  <h2>Results</h2>
</div>
<div class="section-content">
    <p>Dataset utilisé : <a href="https://www.kaggle.com/datasets/proutkarshtiwari/adni-images-for-alzheimer-detection">ADNI Dataset for alzeheimer detection</a></p>
    <p></p>
  
</div>
<div class="section">
<div class="section-title">
  <h2>Analysis and Interpretation</h2>
</div>
<div class="section-content">
  <p>[Detailed analysis of results, scientific interpretation, and implications]</p>
</div>
</div>
<div class="section">
<div class="section-title">
  <h2>Difficulties Encountered</h2>
</div>
<div class="section-content">
  <ul>
    <li><strong>Problem 1:</strong> For the initial tests, the dataset was too limited.</li>
    <li><strong>Problem 2:</strong> Many compatibility issues between Python libraries</li>
  </ul>
</div>
</div>
<div class="section">
<div class="section-title">
  <h2>Next Steps</h2>
</div>
<div class="section-content">
  <ul>
    <li><strong>Objective 1:</strong> [Description of the objective for the following week]</li>
    <li><strong>Objective 2:</strong> [Description of the objective for the following week]</li>
  </ul>
</div>
</div>
<div class="section">
<div class="section-title">
  <h2>Notes / Improvements</h2>
</div>
<div class="section-content">
  <p>None</p>
</div>
</div>
<div class="section">
<div class="signature">
  <h2>Signature</h2>
  <p>FROEHLY Jean-Baptiste, Friday 11/04/2025</p>
</div>
</div>
</body>
</html>

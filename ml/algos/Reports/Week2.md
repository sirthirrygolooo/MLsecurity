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
      <p><strong>Period:</strong> Week 2 - 14/04/2025 to 18/04/2025</p>
    </div>
  </div>

  <div class="section">
    <div class="section-title">
      <h2>Weekly Objectives</h2>
    </div>
    <div class="section-content">
      <ul>
        <li><strong>Objective 1:</strong> Keep digging on the adversarial subject</li>
        <li><strong>Objective 2:</strong> Implement another type of attack</li>
      </ul>
    </div>
  </div>
  <div class="section">
    <div class="section-title">
      <h2>Results Obtained</h2>
    </div>
    <div class="section-content">
      <p>As a new type of attack, I looked at data poisoning, which seemed to me to be the next logical step, given that we're now attacking training data.</p>
      <h3>Data Poisoning</h3>
      <ul>
        <li><strong>Principle:</strong> Modify the training dataset by introducing falsified or biased data.</li>
        <li><strong>Goal : </strong>
        <ul>
            <li>Making the model less efficient</li>
            <li>Make it produce specific errors (backdoor attacks)</li>
            <li>Bias it in certain predictions</li>
        </ul>
        </li>
        <li><strong>Fonctionnement:</strong>
          <ul>
            <li>Gathering information on the target model</li>
            <li>Creation or modification of malicious data</li>
            <li>Injection into the data pipeline</li>
            <ul>
                <li>Contaminated public database</li>
                <li>User contributions (collaborative systems or federated learning)</li>
                <li>Direct attack</li>
            </ul>
            <li>Training the model on this poisoned data</li>
          </ul>
        </li>
        <li><strong>Caractéristiques:</strong>
          <ul>
            <li>Introduced data often difficult to detect</li>
            <li>Allows you to target whether you want to disrupt the model in general or a particular behaviour.</li>
            <li>Difficult to detect</li>
            <li></li>
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

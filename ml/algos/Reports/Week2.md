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
      font-size: 10px;
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
      <h2>Summary</h2>
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
            <li>Very significant impact, especially in sensitive contexts</li>
          </ul>
        </li>
      </ul>
    <p></p>
    <h3>Defence</h3>
    <p>I firstly tried an easy defence method, as my trigger pattern is easy to detect.</p>
    <p>The method I'm going to try and implement is a common one when it comes to countering a backdoor: Trigger Mitigation</p>
    <ul>
    <p>The idead is to neutralize triggers after training through : </p>
        <li>Deactivation of triggered sensitive neurons</li>
        <li>Reverse Engineering of the trigger (easy in my case according to trigger choice)</li>
    </ul>
    </div>
  </div>
  <div class="section">
    <div class="section-title">
      <h2>Results</h2>
    </div>
    <h3>Attack Test</h3>
    <div class="section-content">
      <p>Example of implementation of backdoor attack on MNIST dataset</p>
      <p>My trigger in this case is a small white square introduced on certain images in the bottom right-hand corner. The instruction is that if this white square is there, the image must be labelled 7.</p>
    </div>
    <div class="img-container"><img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/results/minilab/examples.png" alt="normalMNIST">
    <p>Fig.2 Normal behaviour</p>
    </div>
    <div class="img-container">
    <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/results/minilab/prediction_comparison.png" alt="tirggeredMNIST">
    <p>Fig.1 Behaviour when triggered (prediction written on the top)</p>
    </div>
    <div class="img-container">
    <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/results/minilab/confusion_matrix_clean.png" alt="CM_clean">
    <p>Fig.3 Confusion matrix on clean data</p>
    </div>
    <div class="img-container">
    <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/results/minilab/confusion_matrix_poisoned.png" alt="CM_poisoned">
    <p>Fig.4 Confusion matrix on poisoned data</p>
    </div>
    <p>Output</p>
    <pre>
    <code>
Epoch 1 done, Loss: 0.2486, Accuracy: 92.08%
Epoch 2 done, Loss: 0.0613, Accuracy: 98.07%
Epoch 3 done, Loss: 0.0432, Accuracy: 98.70%
Accuracy sur données propres : 98.60%
Attaque backdoor (trigger → 7) : succès à 100.00%</code>
    </pre>
    </div>
    <h3>Defence test</h3>
    <div class="section-content">
        <p>Implementation of randomization of triggered images and mitigation</p>
        <div class="img-container">
            <img src="" alt="">
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

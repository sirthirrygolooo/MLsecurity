<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
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
    <div class="section-title"><h2>General Information</h2></div>
    <div class="section-content">
      <p><strong>Name:</strong> FROEHLY Jean-Baptiste</p>
      <p><strong>Period:</strong> Week 1 - 07/04/2025 to 11/04/2025</p>
    </div>
  </div>
  <div class="section">
    <div class="section-title"><h2>Weekly Objectives</h2></div>
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
    <div class="section-title"><h2>Activities Completed</h2></div>
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
        <li>Analysis of how FGSM and PGD attacks work</li>
      </ul>
      <h3>Friday</h3>
      <ul>
        <li>Defense mechanisms analysis on adversarial attacks through NDNN and DNN</li>
        <li>Report redaction</li>
      </ul>
    </div>
  </div>
  <div class="section">
    <div class="section-title"><h2>Summary</h2></div>
    <h3>Dataset</h3>
    <p>Original Source : ADNI database - <a href="https://adni.loni.usc.edu/">ADNI Website</a></p>
    <p>Kaggle source : <a href="https://www.kaggle.com/datasets/proutkarshtiwari/adni-images-for-alzheimer-detection/">Dataset link</a> - Author : <a href="https://www.kaggle.com/proutkarshtiwari">Utkarsh</a></p>
    <p>Infos : ~490mo, 20257 images</p>
    <p>This dataset provides a collection of preprocessed MRI brain scan images from the ADNI (Alzheimer's Disease Neuroimaging Initiative) project</p>
    <h4>Dataset structure</h4>
    <p>The images are arranged and classified in different categories. These images and categories are referenced in a file <code>train.csv</code> with two rows : </p>
    <code>train.csv</code>
    <table>
        <tr>
            <th>id_code (string)</th>
            <th>diagnosis (int)</th>
        </tr>
        <tr>
            <td>AD-3471</td>
            <td>4</td>
        </tr>
        <tr>
            <td>CN-1819</td>
            <td>0</td>
        </tr>
        <tr>
            <td>LMCI-0760</td>
            <td>3</td>
        </tr>
        <tr>
            <td>...</td>
            <td>...</td>
        </tr>
    </table>
    <p>Here are the different classifications and their diagnosis id</p>
    <ul>
        <li>CN - Cognitively Normal : <code>diagnosis=0</code>; 4077 images</li>
        <li>MCI - Mild Cognitive Impairment : <code>diagnosis=1</code>; 4073 images</li>
        <li>EMCI - Early Mild Cognitive Impairment : <code>diagnosis=2</code>; 3958 images</li>
        <li>LMCI - Late Mild Cognitive Impairment : <code>diagnosis=3</code>; 4074 images</li>
        <li>AD - Alzheimer's Disease : <code>diagnosis=4</code>; 4075 images</li>
    </ul>
    <h3>Techniques</h3>
    <div class="section-content">
      <p>I mainly focused on <strong>Fast Gradient Sign Method</strong> et <strong>Projected Gradient Descent</strong>.</p>
      <h3>Fast Gradient Sign Method (FGSM)</h3>
      <ul>
        <li><strong>Principle:</strong> Attack based on gradient, simple but efficient</li>
        <li><strong>How it works:</strong>
          <ul>
            <li>Calculates the gradient of the loss function with respect to the input image</li>
            <li>Add a small perturbation in the direction that maximises the error</li>
            <li>Formula: <code>x_adv = x + ε * sign(∇x J(θ, x, y))</code></li>
            <li>ε = 0.2 for my tests</li>
          </ul>
        </li>
        <li><strong>Caracteristcs:</strong>
          <ul>
            <li>One-shot attack</li>
            <li>Disturbances often visible to the naked eye</li>
            <li>Quick to calculate</li>
          </ul>
        </li>
      </ul>
      <div class="img-container">
        <img src="https://www.tensorflow.org/tutorials/generative/images/adversarial_example.png" alt="FGSM_illustration">
        <p>Illustration of the FGSM principle (source: TensorFlow.org) for ε = 0.007</p>
      </div>
      <h3>Projected Gradient Descent (PGD)</h3>
      <ul>
        <li><strong>Principle:</strong> More powerful iterative version of <strong>FGSM</strong></li>
        <li><strong>How it works:</strong>
          <ul>
            <li>Apply FGSM in several small steps (10 iterations in my tests)</li>
            <li>Formula: <code>x_adv(t+1) = Proj(x_adv(t) + α * sign(∇x J(θ, x_adv(t), y)))</code></li>
            <li>After each step, projection into a field ε</li>
          </ul>
        </li>
        <li><strong>Caracteristics:</strong>
          <ul>
            <li>More sophisticated than FGSM</li>
            <li>More discrete disturbances</li>
            <li>More expensive to calculate</li>
          </ul>
        </li>
      </ul>
      <h3>Defence on DNN</h3>
      <p>Implemented defence is <strong>Adversarial Training</strong></p>
      <p>This is an effective defence which simply consists of training the model with deliberately poisoned images.</p>
      <p>Protects against adversarial evasion attacks such as these</p>
    </div>
  </div>
  <div class="section">
    <div class="section-title"><h2>Results</h2></div>
    <div class="section-content">
      <p>Training and tests carried out locally on Nvidia RTX 4060 Laptop GPU, Python 3.10.0</p>
      <hr>
      <p>Tests : <strong>10 epochs</strong> — <strong>289.79 secondes</strong></p>
      <p>Batch size : <strong>32</strong></p>
        <p><u><strong>Clean Results</strong></u></p>
      <pre><code>
[*] Clean evaluation:
Accuracy: 0.7791
Average inference time per batch: 0.0010 seconds
Classification Report:
              precision    recall  f1-score   support
           0       0.48      0.47      0.47       834
           1       1.00      1.00      1.00       807
           2       0.99      0.99      0.99       778
           3       0.99      0.99      0.99       809
           4       0.47      0.48      0.47       824
    accuracy                           0.78      4052
   macro avg       0.78      0.78      0.78      4052
weighted avg       0.78      0.78      0.78      4052</code></pre>
    <p><u><strong>Attack Implementation</strong></u></p>
    <p>FGSM</p>
    <pre><code>
[*] Attack: FGSM (ε=0.2)
Accuracy: 0.4173
Average attack+inference time per batch: 0.0362 seconds

Classification Report:
              precision    recall  f1-score   support
           0       0.45      0.36      0.40       834
           1       0.47      0.43      0.45       807
           2       0.36      0.43      0.39       778
           3       0.48      0.45      0.46       809
           4       0.36      0.41      0.38       824
    accuracy                           0.42      4052
   macro avg       0.42      0.42      0.42      4052
weighted avg       0.42      0.42      0.42      4052

[TIME] test_evasion_attack executed in 9.34 seconds</code></pre>
    <p>PGD</p>
    <pre><code>
[*] Attack: PGD (ε=0.2, iter=10)
Accuracy: 0.2648
Average attack+inference time per batch: 0.1489 seconds

Classification Report:
              precision    recall  f1-score   support
           0       0.39      0.28      0.33       834
           1       0.29      0.28      0.29       807
           2       0.19      0.30      0.23       778
           3       0.29      0.27      0.28       809
           4       0.24      0.20      0.22       824
    accuracy                           0.26      4052
   macro avg       0.28      0.27      0.27      4052
weighted avg       0.28      0.26      0.27      4052

[TIME] test_evasion_attack executed in 23.73 seconds</code></pre>
    <hr>
    <p>Defences - on DNN - Adversarial Training</p>
    <p>Training : <strong>Ratio : 0.5</strong> — <strong>460.06 seconds</strong> - <strong>15 epochs</strong></p>
    <p><u><strong>Results after Adversarial Training</strong></u></p>
    <p>Global</p>
    <pre><code>
[*] Evaluation under after defense attack:
Accuracy: 0.7853
Average inference time per batch: 0.0019 seconds

Classification Report:
              precision    recall  f1-score   support
           0       0.47      0.30      0.36       834
           1       1.00      1.00      1.00       807
           2       1.00      1.00      1.00       778
           3       1.00      1.00      1.00       809
           4       0.48      0.66      0.56       824
    accuracy                           0.79      4052
   macro avg       0.79      0.79      0.78      4052
weighted avg       0.78      0.79      0.78      4052

[TIME] evaluate_model executed in 8.15 seconds</code></pre>
    <p>FGSM</p>
    <pre><code>
[*] Attack: FGSM (after defense)
Accuracy: 0.7823
Average attack+inference time per batch: 0.0432 seconds

Classification Report:
              precision    recall  f1-score   support
           0       0.46      0.30      0.36       834
           1       1.00      1.00      1.00       807
           2       1.00      1.00      1.00       778
           3       1.00      1.00      1.00       809
           4       0.48      0.64      0.55       824
    accuracy                           0.78      4052
   macro avg       0.79      0.79      0.78      4052
weighted avg       0.78      0.78      0.78      4052

[TIME] test_evasion_attack executed in 11.51 seconds</code></pre>
    <p>PGD</p>
    <pre><code>
[*] Attack: PGD (after defense)
Accuracy: 0.7813
Average attack+inference time per batch: 0.1573 seconds

Classification Report:
              precision    recall  f1-score   support
           0       0.46      0.33      0.39       834
           1       0.99      1.00      1.00       807
           2       1.00      0.99      1.00       778
           3       1.00      1.00      1.00       809
           4       0.48      0.61      0.54       824
    accuracy                           0.78      4052
   macro avg       0.79      0.79      0.78      4052
weighted avg       0.78      0.78      0.78      4052

[TIME] test_evasion_attack executed in 25.69 seconds</code></pre>
    <p><u><strong>Summary</strong></u></p>
    <pre><code>
Accuracy Metrics:
Initial clean accuracy: 0.7791
Accuracy under FGSM attack: 0.4173 (Drop: 0.3618)
Accuracy under PGD attack: 0.2648 (Drop: 0.5143)
Clean accuracy after defense: 0.7853
Accuracy under FGSM after defense: 0.7823 (Improvement: 0.3650)
Accuracy under PGD after defense: 0.7813 (Improvement: 0.5165)

Performance Metrics:
Standard training time: 289.79 seconds
Adversarial training time: 460.06 seconds (58.76% increase)
Average clean inference time: 0.0010 seconds per batch
Average FGSM attack+inference time: 0.0362 seconds per batch
Average PGD attack+inference time: 0.1489 seconds per batch</code></pre>
    <hr>
    <h3>Images</h3>
    <div class="img-container">
      <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/V1/results/img/4V2/fgsm_attack_full.png" alt="Adversarial Example">
      <p>Fig.1 FGSM pour ε=0.2</p>
    </div>
    <div class="img-container">
      <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/V1/results/img/4V2/pgd_attack_full.png" alt="Adversarial Example">
      <p>Fig.2 PGD pour ε=0.2 et iter=10</p>
    </div>
    <div class="img-container">
      <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/V1/results/img/4V2/attack_comparison.png" alt="Adversarial Example">
      <p>Fig.3 Matrices de confusion pre et post attaques</p>
    </div>
    <p>Mise en place des défenses</p>
    <div class="img-container">
      <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/V1/results/img/4V2/training_time_comparison.png" alt="Adversarial Example">
      <p>Fig.4 Comparaison des temps d'entrainement</p>
    </div>
    <div class="img-container">
      <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/V1/results/img/4V2/defense_comparison.png" alt="Adversarial Example">
      <p>Fig.5 Efficacité des défenses</p>
    </div>
  </div>
  <div class="section">
    <div class="section-title"><h2>Analysis and Interpretation</h2></div>
    <div class="section-content">
      <h3>Initial model performance</h3>
      <h4>Clean datas</h4>
      <ul>
          <li>Accuracy : 77.91%</li>
          <li>Very good performance in classes 1, 2 and 3 (f1-score ~ 1.00)</li>
          <li>Poor performance in classes 0 and 4 (f1-score ~ 0.47)</li>
          <ul>
            <li>These classes are often <strong>confused with each other</strong> as shown by the confusion matrix (c.f. fig. 3) pre-attack: class 0 is often predicted as 4 and vice versa</li>
            <li>This corresponds to the final diganostics of Cognitively Normal (0) and Alzheimer's Disease (4).</li>
          </ul>
      </ul>
      <h4>Under FGSM attack</h4>
      <ul>
        <li>Accuracy : 41.73% → <strong>loss of 36.18 points</strong></li>
        <li>Significant deterioration, particularly for classes 0 and 4</li>
        <li>Widely dispersed confusion (mixture of all classes)</li>
        <li>Attack disrupts initial separation, especially for ‘weak’ classes</li>
      </ul><h4>Under PGD attack</h4>
      <ul>
        <li>Accuracy : 26.48% → <strong>loss of 51.43 points</strong></li>
        <li>PGD is much more powerful and effective than FGSM</li>
        <li>Strong confusion on all classes, prediction seems close to random</li>
        <li>Confusion matrix (c.f. fig 3) shows a very flat, mixed distribution</li>
      </ul>
      <h3>After Adversarial Training</h3>
        <h4>On clean datas</h4>
        <ul>
          <li>Accuracy : 78.53% → slight improvement on the original (77.91%)</li>
          <li>Classes 1, 2 and 3 remain perfect</li>
          <li>Class 4 improves (recall 0.48 → 0.66)</li>
          <li>Class 0 remains a little weak (recall = 0.30)</li>
        </ul>
        <h4>Resistance to attack</h4>
        <ul>
            <li><strong>FGSM after defences : 78.23%</strong></li>
            <ul>
                <li>Almost no impact → the model becomes robust</li>
                <li>Performance similar to that without attack</li>
            </ul>
            <li><strong>PGD after defences : 78.13%</strong></li>
            <ul>
                <li>Impressive resistance with a gain of <strong>+51.65 points</strong> compared to before defence</li>
                <li>Reliable classes (0 and 4) remain sensitive, but overall performance is holding up well</li>
            </ul>
        </ul>
        <h4>System performance</h4>
        <table>
            <tr>
                <th>Criteria</th>
                <th>Standard</th>
                <th>Adversarial</th>
            </tr>
            <tr>
                <td>Training time</td>
                <td>289.79s</td>
                <td>460.06s → +59%</td>
            </tr>
            <tr>
                <td>Inference per batch</td>
                <td>0.0010s</td>
                <td>0.0019s</td>
            </tr>
            <tr>
                <td>Attack + inference FGSM</td>
                <td>0.0362s</td>
                <td>0.0432s</td>
            </tr>
            <tr>
                <td>Attack + inference PGD</td>
                <td>0.1489s</td>
                <td>0.1573s</td>
            </tr>
        </table>
        <p>Adversarial Training <strong>increases robustness at the cost of a longer training time, but inference remains fast</strong></p>
        <h4>Conclusion</h4>
        <ul>
            <li>The basic model is highly accurate but vulnerable to adversarial attacks, particularly PGD.</li>
            <li>After adversarial training, the model becomes more robust while retaining good accuracy on the clean data.</li>
            <li>The trade-off in training time is clearly worth it, especially given the area covered by this dataset.</li>
        </ul>
    </div>
  </div>
<div class="section">
    <div class="section-title"><h2>Difficulties Encountered</h2></div>
    <div class="section-content">
      <ul>
        <li><strong>Problem 1:</strong> Initial dataset was too limited</li>
        <li><strong>Problem 2:</strong> Compatibility issues between Python libraries</li>
      </ul>
    </div>
  </div>
  <div class="section">
    <div class="section-title"><h2>Next Steps</h2></div>
    <div class="section-content">
      <ul>
        <li><strong>Objective 1:</strong>Digging deeper into attack and defence mechanisms, changing parameters, seeing the limits</li>
        <li><strong>Objective 2:</strong>Implemtation of defence on NDNN</li>
        <li><strong>Objective 3:</strong>Looking at other types of attack</li>
      </ul>
    </div>
  </div>
  <div class="section">
    <div class="signature">
      <p>FROEHLY Jean-Baptiste, Friday 11/04/2025</p>
    </div>
  </div>
</body>
</html>
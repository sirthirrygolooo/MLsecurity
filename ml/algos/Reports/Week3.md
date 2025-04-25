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
      <p><strong>Period:</strong> Week 3 - 21/04/2025 to 25/04/2025</p>
    </div>
  </div>
  <div class="section">
    <div class="section-title"><h2>Weekly Objectives</h2></div>
    <div class="section-content">
      <ul>
        <li><strong>Objective 1:</strong> Focus on attacks and understand how they work</li>
        <li><strong>Objective 2:</strong> Continue to implement labs on my image datasets</li>
        <li><strong>Objective 3:</strong> Benchmark the various attacks</li>
      </ul>
    </div>
  </div>
  <div class="section">
    <div class="section-title"><h2>Summary</h2></div>
    <h3>Dataset</h3>
    <p><strong>Special point</strong>: The dataset I was using has been removed from kaggle. So I found another one, with more images, but which were not prepared like the previous one.</p>
    <p>Source : ADNI database - <a href="https://adni.loni.usc.edu/">ADNI Website</a></p>
    <p>New kaggle source : <a href="https://www.kaggle.com/datasets/abdullahtauseef2003/adni-4c-alzheimers-mri-classification-dataset">Dataset link</a> - Author : <a href="https://www.kaggle.com/abdullahtauseef2003">Abdullah Tauseef</a></p>
    <p>Infos : ~490mo, +34.0k images</p>
    <p>This dataset provides a collection of preprocessed MRI brain scan images from the ADNI (Alzheimer's Disease Neuroimaging Initiative) project</p>
    <h4>Dataset structure</h4>
    <p>The images are arranged and classified in different categories in four directories. So I made a script to rearrange them like the previous dataset: </p>
    <p>Pictures are referenced in a <code>train.csv</code> file and are renamed according to their diagnosis (ex: AD-0001.jpg, LMCI-2890 stand for <code>diagnosis_label-number_of_image</code>)</p>
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
        <li>CN - Cognitively Normal : <code>diagnosis=0</code>; 6464 images</li>
        <li>EMCI - Early Mild Cognitive Impairment : <code>diagnosis=1</code>; 9600 images</li>
        <li>LMCI - Late Mild Cognitive Impairment : <code>diagnosis=2</code>; 8960 images</li>
        <li>AD - Alzheimer's Disease : <code>diagnosis=3</code>; 8960 images</li>
    </ul>
    <h3>Attack studied</h3>
    <div class="section-content">
      <p>I mainly focused on <strong>Avdersarial Attacks</strong> and more specifically on <strong>evasion attacks</strong>.</p>
      <p><strong>General Definition</strong> : An <strong>adversarial attack</strong> is an attempt to manipulate a machine learning model via specifically calculated perturbations in the input data.</p>
      <p><strong>Evasion attacks</strong> : Evasion attacks target the model <strong>at the time of inference</strong>, without affecting the training. The aim is to slightly modify the input <code>(x)</code> into a version <code>(x′)</code> so that the model is wrong, while keeping the modifications imperceptible.</p>
      <h3>How evasion attacks work</h3>
      <h4>Principle</h4>
      <p>We seek to construct an input <code>(x′)</code> such that : </p>
      <p><code> f(x′) ≠ f(x) et ∥x−x′∥ < ϵ</code></p>
      <p>where <code>(ϵ)</code> is a tolerated disturbance threshold, and <code>(f)</code> is the model.</p>
      <h4>Attack scenarios</h4>
      <ul>
        <li><strong>White box</strong> : the attacker knows the model <code>(f)</code>, the weights <code>(θ)</code>, and the loss function <code>(J)</code>.</li>
        <li><strong>Black box</strong> : the attacker only has access to the model's outputs.</li>
        <li><strong>Grey box</strong> : the attacker has partial knowledge (for example, access to gradients but not to exact weights</li>
      </ul>
      <h3>Evasion attack types</h3>
      <p>In general, there are 4 main types of evasion attack</p>
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
        <h3>Carlini & Wagner (C&W)</h3>
        <ul>
          <li><strong>Principle:</strong> Optimization-based attack designed to bypass many existing defenses</li>
          <li><strong>How it works:</strong>
            <ul>
              <li>Defines the attack as an optimization problem: find the smallest perturbation <code>(δ)</code> such that the perturbed input is misclassified</li>
              <li>Minimizes a custom loss function that balances the size of the perturbation and the likelihood of misclassification</li>
              <li>Often uses L2 norm but can also be adapted to L0 or L∞</li>
              <li>Formula (L2 norm): <code> min∥x−x′∥22 +c⋅f(x′)</code></li>
            </ul>
          </li>
          <li><strong>Characteristics:</strong>
            <ul>
              <li>Highly effective and stealthy (very small perturbations)</li>
              <li>Bypasses defensive distillation and other robust models</li>
              <li>Computation-heavy due to optimization process</li>
            </ul>
          </li>
        </ul>
        <h3>DeepFool</h3>
        <ul>
          <li><strong>Principle:</strong> Iterative attack that assumes the model is approximately linear around the input</li>
          <li><strong>How it works:</strong>
            <ul>
              <li>Computes the minimal perturbation required to cross the decision boundary</li>
              <li>At each step, approximates the classifier by a linear model and moves toward the closest boundary</li>
              <li>Stops when the label changes</li>
            </ul>
          </li>
          <li><strong>Characteristics:</strong>
            <ul>
              <li>Produces minimal and often imperceptible perturbations</li>
              <li>Works well for untargeted attacks</li>
              <li>Relatively fast and effective</li>
            </ul>
          </li>
        </ul>
        <h3>Black Box attacks</h3>
        <ul>
          <li><strong>Principle:</strong> Attacks where the attacker has no access to model internals (weights, gradients, etc.)</li>
          <li><strong>How it works:</strong>
            <ul>
              <li>Uses query-based approaches: observe outputs for various inputs to infer model behavior</li>
              <li>Can also train a surrogate model using the outputs of the target model, and then attack the surrogate</li>
              <li>Examples include Transfer Attacks, Zeroth-order Optimization (ZOO), and Boundary Attacks</li>
            </ul>
          </li>
          <li><strong>Characteristics:</strong>
            <ul>
              <li>More realistic in real-world scenarios</li>
              <li>Usually require a large number of queries</li>
              <li>Less efficient than white-box attacks but still dangerous</li>
            </ul>
          </li>
        </ul>
        <h3>Common defenses</h3>
        <p>Adversarial Training, Defensive distillation, Gaussian Noise/Dropout, Gradient Smoothing, Adversarial Examples detection, randomization, etc...</p>
        <h3>Common Usages & Impacts</h3>
        <ul>
            <li><strong>Facial Recognition</strong>:  modified glasses that can fool a system.</li>
            <li><strong>Autonomous vehicles</strong>: disruption of road signs</li>
            <li><strong>Biometrics</strong>: false fingerprints.</li>
        </ul>
        <h4>Examples</h4>
        <ul>
            <li>Attack on Google Cloud Vision: a modified panda becomes a gibbon.</li>
            <li>Masks with special patterns to fool facial recognition</li>
        </ul>
    </div>
  </div>
  <div class="section">
    <div class="section-title"><h2>Results</h2></div>
    <div class="section-content">
      <p>Training and tests carried out locally on Nvidia RTX 4060 Laptop GPU, Python 3.10.0</p>
      <hr>
      <p><strong>20 epochs</strong> — Optimizer Adam - CNN : 2*conv+pooling, 2 fully conn., fn: RelU/p>
      <p>Batch size : 32</p>
      <p>Avg Epoch Time : 75s</p>
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
        <h3>Initial Model Performance</h3>
        <h4>Clean Data</h4>
        <ul>
          <li>Accuracy: 81.24%</li>
          <li>Very strong performance on class 0 (f1-score = 0.93)</li>
          <li>Consistent results across other classes (f1-scores between 0.75 and 0.81)</li>
          <li>Macro average f1-score: 0.82 → well-balanced model</li>
        </ul>
        <h3>Impact of Adversarial Attacks</h3>
        <h4>FGSM (ε = 0.2)</h4>
        <ul>
          <li>Accuracy: 10.81% → <strong>drop of 70.43 points</strong></li>
          <li>Severe performance degradation across all classes (f1-scores around 0.1)</li>
          <li>The model is easily fooled, predictions become nearly random</li>
          <li>Attack + inference time: 0.0724s per batch</li>
        </ul>
        <h4>PGD (ε = 0.2, 10 iterations)</h4>
        <ul>
          <li>Accuracy: 8.81% → <strong>drop of 72.43 points</strong></li>
          <li>Even more destructive than FGSM</li>
          <li>The model completely loses its generalization ability</li>
          <li>Attack + inference time: 0.3330s per batch (about ×200 slower than clean inference)</li>
        </ul>
        <h3>Conclusion</h3>
        <ul>
          <li>The model performs excellently on clean data</li>
          <li>But it is highly vulnerable to adversarial attacks, even basic ones like FGSM</li>
          <li>PGD shows that minimal perturbations can entirely derail the model</li>
          <li>This analysis highlights the importance of considering security in deep learning systems</li>
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
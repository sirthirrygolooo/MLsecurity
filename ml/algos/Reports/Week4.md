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
      <p><strong>Period:</strong> Week 4 - 28/04/2025 to 02/05/2025</p>
    </div>
  </div>
  <div class="section">
    <div class="section-title"><h2>Weekly Objectives</h2></div>
    <div class="section-content">
      <ul>
        <li><strong>Objective 1:</strong> Improving the precision of attacks: efficiency and undetectability </li>
        <li><strong>Objective 2:</strong> Further work on deepfool</li>
        <li><strong>Objective 3:</strong> Still the idea of benchmarking various attacks</li>
      </ul>
    </div>
  </div>
  <div class="section">
    <div class="section-title"><h2>Summary</h2></div>
    <h3>Dataset - Still the same</h3>
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
          <li><strong>Principle:</strong> Gradient-based iterative attack assuming local linearity of the model</li>
          <li><strong>How it works:</strong>
            <ul>
              <li>Approximates the classifier as a linear model around the current input</li>
              <li>At each step, computes the minimal perturbation needed to reach the closest decision boundary</li>
              <li>Moves slightly toward that boundary and updates the input</li>
              <li>Stops when the classifier changes its prediction</li>
              <li>Formally solves: find <code>r</code> such that <code>argmax f(x + r) ≠ argmax f(x)</code>, with minimal <code>||r||</code></li>
            </ul>
          </li>
          <li><strong>Characteristics:</strong>
            <ul>
              <li>Iterative and relatively fast</li>
              <li>Perturbations are minimal and usually imperceptible</li>
              <li>Non-targeted attack (aims to cause any misclassification)</li>
              <li>Requires access to model gradients (white-box attack)</li>
              <li>More precise than FGSM but computationally more expensive</li>
            </ul>
          </li>
        </ul>
        <div class="img-container">
          <img src="https://www.researchgate.net/publication/369332060/figure/fig2/AS:11431281177028028@1690336665579/Illustration-of-DeepFool-attack-algorithm.png" alt="DeepFool illustration">
          <p>Illustration of DeepFool attack algorithm</p>
        </div>
        <h4>Explanation of the Scheme</h4>
        <p>
          The illustration above represents the DeepFool attack algorithm in action. Here’s a breakdown of the key elements:
        </p>
        <ul>
          <li><strong>Decision Boundaries (Boundary1, Boundary2, Boundary3):</strong>
            The gray curved lines represent the decision boundaries of a classifier, which separate different classes in the input space. These boundaries are where the classifier's prediction changes.
          </li>
          <li><strong>Initial Input <code>x_0</code>:</strong>
            The point <code>x_0</code> (at the center-bottom) is the original input, correctly classified by the model.
          </li>
          <li><strong>Perturbation (Red Arrow):</strong>
            The red arrow indicates the minimal perturbation applied to <code>x_0</code>> to move it across a decision boundary. DeepFool calculates this perturbation iteratively to ensure it is as small as possible.
          </li>
          <li><strong>Perturbed Input <code>x^</code>:</strong>
            The point <code>x^</code> is the perturbed input, now located on the other side of the decision boundary (Boundary3), resulting in misclassification by the model.
          </li>
          <li><strong>Search Region (Green Triangle):</strong>
            The green dashed triangle represents the region where DeepFool explores to find the minimal perturbation. It iteratively adjusts the input to approach the nearest decision boundary.
          </li>
        </ul>
        <p>
          This visualization highlights how DeepFool efficiently finds the smallest perturbation needed to fool the classifier, making it a powerful tool for generating adversarial examples.
        </p>
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
      <p><strong>20 epochs</strong> — Optimizer Adam or SGD (learning rate : 0.004, momentum : 0.9) - CNN : 2*conv+pooling, 2 fully conn., activation function: RelU</p>
      <p>Batch size : 32</p>
      <p>Avg Epoch Time : 90.67s</p>
      <p><u><strong>Clean Results</strong></u></p>
      <pre><code>
[*] Clean evaluation:
Accuracy: 0.8719
Average inference time per batch: 0.0020 seconds

Classification Report:
              precision    recall  f1-score   support
           0       0.97      0.99      0.98      1295
           1       0.89      0.84      0.87      1947
           2       0.83      0.78      0.81      1765
           3       0.82      0.90      0.86      1790
    accuracy                           0.87      6797
   macro avg       0.88      0.88      0.88      6797
weighted avg       0.87      0.87      0.87      6797

[TIME] evaluate_model executed in 24.48 seconds</code></pre>
    <p><u><strong>Attack Implementation</strong></u></p>
    <p>FGSM</p>
    <pre><code>
[*] Attack: FGSM (ε=0.02)
Accuracy: 0.7138
Average attack+inference time per batch: 0.0707 seconds

Classification Report:
              precision    recall  f1-score   support
           0       0.93      0.86      0.90      1295
           1       0.74      0.65      0.69      1947
           2       0.58      0.65      0.61      1765
           3       0.69      0.74      0.71      1790
    accuracy                           0.71      6797
   macro avg       0.74      0.73      0.73      6797
weighted avg       0.72      0.71      0.72      6797

[TIME] test_evasion_attack executed in 29.91 seconds</code></pre>
    <p>PGD</p>
    <pre><code>
[*] Attack: PGD (ε=0.02, iter=10)
Accuracy: 0.6955
Average attack+inference time per batch: 0.3331 seconds

Classification Report:
              precision    recall  f1-score   support
           0       0.92      0.86      0.89      1295
           1       0.72      0.63      0.67      1947
           2       0.56      0.62      0.59      1765
           3       0.67      0.72      0.69      1790
    accuracy                           0.70      6797
   macro avg       0.72      0.71      0.71      6797
weighted avg       0.70      0.70      0.70      6797

[TIME] test_evasion_attack executed in 82.16 seconds</code></pre>
    <hr>
    <pre><code>
Accuracy Metrics:
Initial clean accuracy: 0.8719
Accuracy under FGSM attack: 0.7138 (Drop: 0.1580)
Accuracy under PGD attack: 0.6955 (Drop: 0.1764)
Accuracy under DeepFool attack: 0.1487 (Drop: 0.7231)

Performance Metrics:
Standard training time: 1906.31 seconds
Average clean inference time: 0.0020 seconds per batch
Average FGSM attack+inference time: 0.0707 seconds per batch
Average PGD attack+inference time: 0.3331 seconds per batch
Average DeepFool attack+inference time: 5.4564 seconds per batch</code></pre>
    <hr>
    <h3>Images</h3>
    <div class="img-container">
      <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/results-archive/img/attacks/attack_visualization.png" alt="attack_comparison">
      <p>Fig.0 Attack comparizon for ε=0.4</p>
    </div>
    <div class="img-container">
      <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/results-archive/img3/attacks/attack_visualization.png" alt="attack_comparison2">
      <p>Fig.1 Attack comparison for ε=0.02</p>
    </div>
    <div class="img-container">
      <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/results-archive/img3/attacks/attack_example_0.png" alt="attack_example">
      <p>Fig.2 Attack comparison for ε=0.02 with DeepFool</p>
    </div>
    <div class="img-container">
      <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/results-archive/img3/attacks/attack_example_4.png" alt="attack_example2">
      <p>Fig.3 Attack comparison for ε=0.02 with DeepFool other example</p>
    </div>
    <div class="img-container">
      <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/results-archive/img3/performance_comparison.png" alt="Accuracy comparison">
      <p>Fig.4 Accuracy comparison</p>
    </div>
    <div class="img-container">
      <img src="https://raw.githubusercontent.com/sirthirrygolooo/MLsecurity/refs/heads/master/ml/algos/ARTLab/results-archive/img3/attack_comparison.png" alt="Confusion Matrix">
      <p>Fig.5 Confusion Matrix comparison</p>
    </div>
    <div class="img-container">
      <img src="https://github.com/sirthirrygolooo/MLsecurity/blob/master/ml/algos/ARTLab/results-archive/img3/training_metrics.png?raw=true" alt="training_metrics">
      <p>Fig.6 Training Metrics (20 epochs)</p>
    </div>
    <div class="section">
      <div class="section-title"><h2>Analysis and Interpretation</h2></div>
      <div class="section-content">
        <p></p>
        <p>We can see that it's very easy to obtain invisble perturbation for Human eye with FGSM and PGD with an interessant efficiency of attack.</p>
        <p>We can also see that DeepFool, at least with the parameters used so far, is not as discreet as expected on grayscale images of this type. It's undeniably extremely effective, but the difference is easily distinguishable </p>
        <h3>Initial Model Performance</h3>
        <h4>Clean Data</h4>
        <ul>
          <li>Accuracy: 87.19%</li>
          <li>Excellent performance on class 0 (f1-score = 0.98)</li>
          <li>Balanced results across other classes with f1-scores between 0.81 and 0.87</li>
          <li>Macro average f1-score: 0.88 → indicates good generalization across classes</li>
          <li>Fast inference: 0.0020 seconds per batch</li>
        </ul>
        <h3>Impact of Adversarial Attacks</h3>
        <h4>FGSM (ε = 0.02)</h4>
        <ul>
          <li>Accuracy: 71.38% → <strong>drop of 15.80 points</strong></li>
          <li>Significant decline in performance on classes 1–3 (f1-scores range from 0.61 to 0.71)</li>
          <li>Class 0 remains relatively robust (f1-score = 0.90)</li>
          <li>Attack + inference time: 0.0707 seconds per batch (×35 slower than clean)</li>
        </ul>
        <h4>PGD (ε = 0.02, 10 iterations)</h4>
        <ul>
          <li>Accuracy: 69.55% → <strong>drop of 17.64 points</strong></li>
          <li>More impactful than FGSM, especially on classes 1 and 2</li>
          <li>f1-scores drop further (as low as 0.59 for class 2)</li>
          <li>Attack + inference time: 0.3331 seconds per batch (×166 slower than clean)</li>
        </ul>
        <h4>DeepFool</h4>
        <ul>
          <li>Accuracy: 14.87% → <strong>drop of 72.31 points</strong></li>
          <li>Catastrophic degradation — the model fails to make meaningful predictions</li>
          <li>Highlights extreme sensitivity to subtle, well-crafted perturbations</li>
          <li>Attack + inference time: 5.4564 seconds per batch (over ×2700 slower than clean inference)</li>
        </ul>
        <h3>Conclusion</h3>
        <ul>
          <li>The model performs very well on clean data, with high and balanced accuracy</li>
          <li>However, it is vulnerable to even low-magnitude adversarial attacks (ε = 0.02)</li>
          <li>PGD and especially DeepFool demonstrate how easily the model's performance can collapse</li>
          <li>These results emphasize the necessity of integrating adversarial robustness into model development pipelines</li>
        </ul>
      </div>
    </div>
<div class="section">
    <div class="section-title"><h2>Difficulties Encountered</h2></div>
    <div class="section-content">
      <ul>
        <li><strong>Problem 1:</strong> I had problems with C&W : the execution takes far too long, I've never been able to afford to finish one, so either I have to completely revise the parameters, 
or I have to abandon the idea of implementing this attack.</li>
        <li><strong>Problem 2:</strong> DeepFool doesn't behave very well on my images, so I think I need to review my settings in detail to try other things. 
I'm not sure whether it's a question of the dataset or the parameters, given that on paper,
according to the maths behind it, there's no reason why you can't get interesting and effective results.</li>
      </ul>
    </div>
  </div>
  <div class="section">
    <div class="signature">
      <p>FROEHLY Jean-Baptiste, Friday 02/05/2025</p>
    </div>
  </div>
</body>
</html>
# Learning from Synthetic Data: A Deep Dive into Tabular Classification at Scale

## 1. Project Overview
This project presents a comprehensive investigation into the **generation and application of synthetic data** for tackling **complex, imbalanced tabular classification problems**.  

The primary goal is to move beyond traditional oversampling techniques such as **SMOTE** and systematically explore how **modern, model-driven, and algorithmic approaches** can generate high-quality synthetic datasets.  

By benchmarking multiple strategies across different scales, the project aims to answer key questions:  
- Can synthetic data improve model generalization?  
- How do different generation methods compare in terms of scalability, performance, and resource efficiency?  
- What trade-offs exist between synthetic, oversampled, and original datasets?  

This project systematically evaluates the impact of different data generation strategies on model performance, robustness, data quality, and algorithmic scalability. We compare models trained on four distinct datasets:

- A perfectly balanced dataset (theoretical benchmark).
- A realistically imbalanced dataset (the problem).
- A dataset augmented with the classifical SMOTE algorithm.
- A large scale dataset created using a novel, model driven filtering technique.

 The analysis culminates in a rigous head to head comparison, not just on predictive accuracy, but on the engineering critical axes of algorithmic efficienct and production readiness.

## 2. Motivation: Why This Project Matters
Class imbalance is a persistent and challenging problem in real world machine learning applications, from fraud detection to medical diagnosis. While simple oversampling techniques like SMOTE are common, they often fail on complex datasets where the underlying data distribution is naunced.

This project was motivated by the need to answer a deeper question: **Can we learn the underlying patterns of a minority class from imbalanced data and use that knowledge to generate high-quality, realistic synthetic samples that lead to more robust and generalisable models?**

This work stands at the intersection of Machine Learning, Data Structures & Algorithms (DSA), and Systems Design, demonstrating a holistic approach to solving a data centric problem with an engineering mindset.

## 3. Dataset Design
A custom, complex tabular dataset was synthetically generated to provide a challanging and controllable environment for this investigation.

| Dataset Version    | Rows       | Features              | Class Balance | Purpose                                                                 |
|--------------------|------------|-----------------------|---------------|-------------------------------------------------------------------------|
| Original Balanced  | ~128,000   | 40 (30 Num, 10 Cat)  | 50% / 50%     | Establish a theoretical "gold standard" performance benchmark.          |
| Imbalanced         | ~70,000    | 40 (30 Num, 10 Cat)  | ~92% / 8%     | Simulate a realistic, challenging production scenario.                  |
| Synthetic Datasets | ~300,000+  | 40 (30 Num, 10 Cat)  | ~50% / 50%    | Provide a balanced, rich training set for the final models.             |

## 4. Project Phases & methodology
This project was executed in a series of logical phases, where the insights from each step directly motivated the next.

## Phase 1 & 2: Baseline and the Failure of Classical Methods
- **Goal:** Quantify the problem and test the standard industry solution.
- **Actions:**
    - An H2O AutoML model was trained on the **Balanced** data, achieving a gold standard AUPRC of **0.963**.
    - The same model trained on the **imbalanced** data saw its AUPRC collapse to **0.741**, demonstrating the severity of the problem. This model also produced brittle, untrustworthy metrics (100% precision/recall).
    - The **SMOTE** algorithm was applied to the imbalanced training set.
- **Key insight:** Training on the SMOTE augmented data **failed to improve performance (AUPRC: 0.734).** This proved that for complex data, naive geometric oversampling is insufficient and can be detrimental.

## Phase 3 & 4: A Novel Model Driven Solution
- **Goal:** Develop a more intelligent synthetic data generation technique.
- **Actions:**
    - A high performing model (H2O AutoML Leader) was trained on the imbalanced data to learn its underlying patterns.
    - A batch optimised pipeline was engineered to generate millions of candidate samples by interpolating between real minority points.
    - The trained model was used as a **"quality filter"**: Only candidate samples that the model predicted as high confidence minority class were accepted. This process yielded **~250,000 high quality synthetic samples**
- **Key insight:** The final model trained on this new, rich dataset produced more realistic and trustworthy performance metrics, suggesting it had learned a more nuanced and generalisable decision boundary.

## Phase 5. Visualising Data Quality with Latent Space Analysis
- **Goal:** Create definitive, visual proof of our synthetic data's quality.
- **Actions:**
    - A PyTorch Autoencoder was trained to project the high dimensional data into a 2D latent space.
    - Original, SMOTE, and Model Driven synthetic data points were projected and visualised using t-SNE.
- **Key insight:** The visualisation was damning for SMOTE. It clearly showed SMOTE generating samples in unrealistic "No-Man's-Land" regions between true data clusters. In contrast, our model driven samples organically expanded and densified the existing, real minority clusters, proving their superior quality and realism.
<div style="margin-top: 25px;"></div>

<figure>
  <img src="plots/tsne.png" alt="t-SNE Visualization" width="500">
  <figcaption><i>t-SNE plot showing Original Minority (blue), Model Driven (green) respecting the data manifold and SMOTE (orange) generating noise in between.</i></figcaption>
</figure>

## Phase 6 & 7: Algorithms Alternatives and Robustness Testing
- **Goal:** Explore other generation methods and test the real world robustness of our models.
- **Actions:**
    - A graph based generation methods and test the real world robustness of our models.
    - The champion models (imbalanced, SMOTE, ModelDriven) were evaluated on a "perturbed" test set where random noise was added to simulate data drift.

- **Key insight:** A counter intuitive but powerful result emerged. The ModelDriven model showed that highest sensitivity to noise. I concluded this was not a weakness, but a sign of its **sophistication**. It had learned a complex function that relied on many subtle features, making it inherently more sensitive to disruptions, unline the "numb" baseline models that had learned overly simplistic rules.

<div style="margin-top: 25px;"></div>

<figure>
  <img src="plots/robust_performance.png" alt="t-SNE Visualization" width="500">
  <figcaption><i>Performance drop on noisy data. The ModelDriven model's sensitivity is a direct result of its higher sophistication.</i></figcaption>
</figure>

## Phase 10: Algorithmic Efficiency & Scalability Analysis
- **Goal:** Analysing the production readiness of each generation method from an engineering prospective.
- **Actions:**
    1. A **theoretical complexity analysis** was performed, concluding that k-NN based methods (SMOTE, Graph-Based) are super linear (`O(N log N)`), while the Model-Driven approach is linear (`O(N)`).
    2. An **empirical benchmark** was conducted, measuring real world time and memory usage.
    3. A discussion on **parallelisation** highlighted that the Model-Driven approach is "embarassingly parallel" and thus perfectly suited for distributed systems like **Spark**.
- **Key insight:** The `ModelDriven` approach is not only superion in terms of data quality and model performance but is also the most **algorithmically efficient and scalable** solution, making it the clear choice for enterprise level deployment.




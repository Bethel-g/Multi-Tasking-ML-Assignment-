# Peer Teaching: Multi-Task Learning (MTL) as a Regularization Technique

## 1. What is Multi-Task Learning?
Multi-Task Learning (MTL) is a regularization technique where one model learns multiple related tasks at the same time. Instead of training separate models for each task, the model shares knowledge between tasks through common hidden layers.

## 2. Why it Works as Regularization
When a model learns several related tasks simultaneously:
* **Avoids Overfitting**: It prevents the model from memorizing noise specific to one task.
* **Shared Representation**: Learning common patterns improves generalization across all tasks.
* **Inductive Bias**: One task helps "teach" another, leading to more robust features.

## 3. How It Works: Architecture
The model consists of two main components:
1. **Shared Layers**: Early layers that learn general features from the input data.
2. **Task-Specific Heads**: Individual output layers for each task.
   * *Example*: Input → Shared Backbone → [Regression Head (Score)] & [Classification Head (Pass/Fail)]

## 4. Bias-Variance Tradeoff
* **Bias**: Error due to overly simple assumptions. High bias leads to **Underfitting**.
* **Variance**: Error due to sensitivity to training data noise. High variance leads to **Overfitting**.
* **MTL's Role**: MTL reduces variance by forcing the model to learn generalized representations that work for multiple objectives, effectively balancing the tradeoff.

## 5. Use Cases & Limitations
### Best Use Cases
* Healthcare diagnosis (Severity + Category)
* Student analytics (Final Grade + Pass/Fail)
* NLP & Computer Vision

### Limitations
* **Relatedness**: Tasks must share some underlying logic.
* **Negative Transfer**: If tasks are unrelated, they can interfere with each other's learning.
* **Loss Balancing**: Determining the importance of each task's loss can be challenging.

---
*Academically Prepared for Machine Learning Assignment 2.*

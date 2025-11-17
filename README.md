# Sentiment Analysis Using BERT

**Date:** April 8, 2025  
**Contributors:**  
- Amina Alisheva  
- Ariana Kenbayeva  
- Jafar Isbarov  

---

## Overview
This project explores sentiment classification for the **Azerbaijani language** using **BERT-based models**. The work consists of two main parts:

1. Analysis of an existing **fine-tuned bert-base-uncased model** trained on the IMDB dataset.  
2. Fine-tuning the **aLLMA-Large Azerbaijani language model** on a labeled dataset of Azerbaijani news articles.

The goal is to evaluate how well transformer-based architectures perform on a **low-resource, agglutinative language** and to determine optimal fine-tuning strategies.

---

## Datasets
- **Azerbaijani News Dataset:**  
  Contains **23,963 news articles** labeled with three sentiment classes: negative (0), neutral (1), positive (2).  
- Dataset used for training, validation, and testing split into:
  - **70%** training  
  - **10%** validation  
  - **20%** testing  
- Data originates from annotated Azerbaijani media texts.

---

## Part 1: Analysis of bert-base-uncased Model
This section focuses on understanding the structure and limitations of the open-source model.

### Key Findings
- **Inputs:**  
  - Input IDs  
  - Attention masks  
  - Token type IDs  
- **Outputs:**  
  Predicts two classes: positive and negative.

- **Token Limit:**  
  Maximum input sequence length is **512 tokens**.

- **Case Sensitivity:**  
  The model is **uncased**, treating uppercase and lowercase forms identically.

- **Suitability for Azerbaijani:**  
  The model can be adapted with fine-tuning, though it does not inherently handle the morphological complexity of agglutinative languages.

---

## Part 2: Fine-Tuning aLLMA-Large for Azerbaijani Sentiment Analysis
The second part evaluates the performance of an Azerbaijani-pretrained transformer.

### Training Details
- Model: **aLLMA-Large**
- Labels: 3 sentiment classes  
- Training time: Approximately **2 hours**
- Model trained for three epochs

### Results Across Epochs

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
|-------|----------------|------------------|----------|-----------|
| 1     | 0.8622         | 0.8054           | 63.39%   | 0.5945    |
| 2     | 0.2714         | 0.6737           | 74.90%   | 0.7459    |
| 3     | 0.2673         | 1.0492           | 73.01%   | 0.7288    |

### Observations
- Peak performance achieved in the **second epoch**.  
- The third epoch exhibits **overfitting**, indicated by rising validation loss.  
- Strong performance attributed to **pretraining on a large Azerbaijani corpus**.

---

## Discussion
The results show that transformer-based models can effectively handle sentiment classification tasks for **low-resource languages** when supported by domain-specific pretraining. The significant improvement from epoch 1 to epoch 2 demonstrates successful adaptation to the dataset. The later overfitting emphasizes the importance of **early stopping** and well-tuned hyperparameters.

The analysis of the English fine-tuned BERT model highlights several limitations when applied directly to Azerbaijani due to linguistic differences, reinforcing the need for Azerbaijani-focused models.

---

## Challenges & Limitations
- Overfitting after the second epoch despite decreasing training loss  
- Morphological richness of Azerbaijani requires tokenizer optimization  
- Limited availability of high-quality labeled data  
- Domain variability in news articles poses difficulty for generalization

---

## Contributions
### Part 1
- **Ariana Kenbayeva:** Analysis of the fine-tuned English BERT model  
- **Amina Alisheva:** Support in analysis and documentation

### Part 2
- **Jafar Isbarov:** Fine-tuning experiments on the Azerbaijani dataset  
- **Ariana Kenbayeva:** Assistance with training setup and evaluation of fine-tuning
---

## Results Summary
- Fine-tuned transformer achieved **74.90% accuracy** and **0.7459 F1 score** on the Azerbaijani dataset.  
- Pretraining on Azerbaijani text significantly improved generalization.  
- The optimal performance was achieved with **two epochs** of training.

---

## Future Improvements
- Expand dataset size and domain coverage  
- Improve tokenizer to better reflect Azerbaijani morphology  
- Experiment with data augmentation and regularization  
- Explore multilingual or cross-lingual training approaches  
- Apply domain-specific pretraining for news articles

---

## References
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*  
- Isbarov, J. et al. (2024). *Open foundation models for Azerbaijani language.*  
- Analytics Vidhya tutorials on BERT fine-tuning  
- aLLMA-Large model documentation on Hugging Face

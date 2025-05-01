# Logistic Regression - Breast Cancer Classification

This project builds a binary classifier to detect breast cancer using logistic regression.

## What is Included:
- Train/Test split
- Standardization of features
- Logistic Regression model training
- Evaluation using:
  - Confusion Matrix
  - Precision, Recall
  - ![image](https://github.com/user-attachments/assets/4b37edfb-ad13-4468-a7ad-e11a526343fd)

  - ROC-AUC Score
  - ROC Curve plotting
- Threshold tuning
- Sigmoid function explanation and plotting

## Tools Used:
- Python
- scikit-learn
- pandas
- matplotlib
- seaborn

## Dataset:
- Breast Cancer Wisconsin dataset (available in `sklearn.datasets`)


Confusion Matrix:
 [[41  2]
 [ 1 70]]

Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.95      0.96        43
           1       0.97      0.99      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114

ROC-AUC Score: 0.9974
Precision: 0.9722
Recall: 0.9859
Optimal Threshold based on TPR-FPR difference: 0.6207

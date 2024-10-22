
# RandomForest Model for Credit Card Fraud Detection

This project uses a **RandomForestClassifier** to detect credit card fraud using a dataset that contains a highly imbalanced set of transactions.

## Project Structure

1. **Data Preprocessing**: 
   - Downsamples the majority (non-fraud) class to maintain a 10:1 ratio of non-fraud to fraud transactions.
   - Features include time-related features like the hour of the transaction and log transformation of the amount to improve distribution.

2. **Model Selection and Evaluation**:
   - **RandomForestClassifier** is trained with varying class weights and threshold values using cross-validation to determine the best-performing model based on F1-score for fraud detection (class 1).
   - Cross-validation is performed using 5 folds.

3. **Grid Search**:
   - **Class Weights**: Varies from 1.01 to 2.11 in steps of 0.01 to handle the imbalance in the dataset.
   - **Threshold**: Ranges from 0.495 to 0.505 in steps of 0.005 to find the optimal classification decision boundary for fraud transactions.

4. **Metrics**:
   - Metrics tracked include F1-score, precision, recall, and ROC-AUC, specifically for fraud detection.

5. **Model Evaluation**:
   - Once the best model is selected through cross-validation, it is evaluated on the test set using the best threshold and class weight.

## Model Training and Performance

### Model Configuration:
- **Number of CPU cores used**: 12
- **Total physical CPU cores available**: 24
- **Cross-validation splits**: 5
- **Number of class weights and threshold configurations tried**: 1665

### Training Duration:
- Overall Model Training and Tuning Start Time: 2024-10-21 21:00:48
- Overall Model Training and Tuning End Time: 2024-10-21 21:06:22
- Training duration: 5 minutes and 34 seconds

### Best Model Configuration:
- **Best Class Weight**: 1.07
- **Best Threshold**: 0.495
- **Best F1-score (Fraud)** from Cross-Validation: 0.90
- **Best Precision (Fraud)** from Cross-Validation: 0.99
- **Best Recall (Fraud)** from Cross-Validation: 0.82
- Model saved at `./models/credit-card-fraud-detection-randomforest.pkl`

### Test Set Performance:
- **Final F1-score (Fraud)** from Test data: 0.93
- **Final Precision (Fraud)** from Test data: 1.00
- **Final Recall (Fraud)** from Test data: 0.87
- **Final Model ROC-AUC Score** from Test data: 0.99

### Final Confusion Matrix:
```
[[985   0]
 [ 13  85]]
```

### Classification Report:
```
               precision    recall  f1-score   support

   Non-Fraud       0.99      1.00      0.99       985
       Fraud       1.00      0.87      0.93        98

    accuracy                           0.99      1083
   macro avg       0.99      0.93      0.96      1083
weighted avg       0.99      0.99      0.99      1083
```

## Project Structure

```
.
├── data/                     # Folder for the dataset
├── models/                   # Saved models and checkpoints
├── plots/                    # Output plots for evaluation
├── credit-card-fraud-detection-by-ml-randomforest.ipynb         # Main script to run the project
├── requirements.txt           # Python dependencies
└── README.md                  # Project description
```

## Visualization

The project includes various visualizations for model performance:

- **Confusion Matrix**
- **ROC Curve**
- **Precision-Recall Curve**
- **Correlation Heatmap** for feature relationships

![Performance Diagrams](./plots/performance-diagrams.png)
![Correlation Heatmap](./plots/Correlation-Heatmap-of-Features.png)


## Conclusion

The best-performing **RandomForestClassifier** model achieved a **Final F1-score of 0.93** for detecting fraud on the test set. While the precision for fraud detection was perfect (1.00), the recall of 0.87 indicates some fraud transactions were missed. The model showed a very strong overall performance with a **ROC-AUC score of 0.99**, indicating that it can distinguish between fraud and non-fraud transactions very effectively.

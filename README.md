# Iris-Classifiers
Iris Classifiers


---

# Iris Dataset Classifier Comparison

This project demonstrates a comprehensive approach to comparing multiple classifiers using the Iris dataset. The objective is to test and tune various machine learning models to evaluate their performance on this classic dataset. Each classifier is trained, tuned using GridSearchCV, and evaluated for accuracy and cross-validation scores.

## Project Overview

The Iris dataset is loaded, preprocessed, and split into training and test sets. A variety of machine learning classifiers are then tuned and evaluated to determine the optimal model for classifying the Iris flower species. The project includes hyperparameter tuning for each classifier using extensive grids, enabling a comparison of performance based on model accuracy and cross-validation metrics.

## Classifiers Used

The following classifiers are explored with custom hyperparameter grids for detailed model comparison:

- Random Forest Classifier
- Decision Tree Classifier
- Perceptron
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (Neural Network)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Bagging Classifier
- AdaBoost Classifier
- Naive Bayes (GaussianNB)
- Gradient Boosting Classifier

Each classifier’s performance is assessed based on test accuracy and cross-validation accuracy. The best hyperparameters for each model are determined and recorded.

## Key Libraries

- **Pandas, NumPy** for data handling
- **Scikit-learn** for classifiers, preprocessing, hyperparameter tuning, and evaluation metrics
- **TensorFlow (Keras)** for neural network modeling
- **GridSearchCV** for hyperparameter tuning
- **Cross-validation** to ensure robust evaluation of model performance

## How to Run

1. Ensure the required libraries are installed.
2. Clone the repository and navigate to the project directory.
3. Run the script to perform data preprocessing, model training, hyperparameter tuning, and evaluation.

## Results

Each model's accuracy and cross-validation mean accuracy are printed upon execution. The script identifies the best model for the Iris dataset based on test accuracy and cross-validation score. 

## Future Work

Further enhancements could include testing additional models, using other datasets, and applying additional feature engineering methods to further improve model accuracy.


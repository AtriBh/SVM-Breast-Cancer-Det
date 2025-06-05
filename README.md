# Breast Cancer Classification with SVM

## Project Overview

This project implements a Support Vector Machine (SVM) classifier on the Wisconsin Breast Cancer dataset to distinguish malignant from benign tumors. The focus was on optimizing recall to minimize false negatives, which is critical in medical diagnostics.

## Key Features

- Implemented SVM with both linear and RBF kernels
- Visualized decision boundaries using PCA
- Conducted hyperparameter tuning for C and gamma
- Performed grid search validation
- Focused on recall metric to minimize false negatives

## Methodology

### Data Preparation
- Standardized features using StandardScaler
- Split data into training and test sets
- Applied PCA for visualization (2 components)

### Model Training
```python
from sklearn.svm import SVC

# RBF Kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma=1.0, random_state=42)
svm_rbf.fit(X_train, y_train)

# Linear Kernel
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train, y_train)
```

### Hyperparameter Tuning
Used validation curves to find optimal parameters:

![Validation Curve for Gamma](Screenshot_2025-06-05_182513.png)
*Gamma validation curve showing optimal value at 1.0*

![Validation Curve for C](Screenshot_2025-06-05_182520.png)
*C validation curve showing optimal value at 1.0*

### Grid Search Implementation
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': np.logspace(-3, 3, 7),
    'gamma': np.logspace(-3, 2, 6),
    'kernel': ['rbf']
}

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='recall_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
```

### Decision Boundary Visualization
![Decision Boundary](Screenshot_2025-06-05_180339.png)
*Decision boundary of SVM on PCA-transformed data*

## Results

| Metric          | RBF Kernel | Linear Kernel |
|-----------------|------------|---------------|
| Training Recall | 0.982      | 0.961         |
| Test Recall     | 0.974      | 0.953         |
| Optimal C       | 1.0        | 1.0           |
| Optimal Gamma   | 1.0        | N/A           |


## Conclusion

The SVM with RBF kernel achieved superior performance (test recall: 0.974) compared to the linear kernel. Through systematic validation and grid search, we confirmed the optimal hyperparameters to be C=1.0 and gamma=1.0. The decision boundary visualization demonstrates effective separation of malignant and benign cases in the reduced PCA space.

```markdown
# Breast Cancer Classification with SVM

![SVM Decision Boundary](Screenshot_2025-06-05_180339.png)

## Project Overview

This project implements a Support Vector Machine (SVM) classifier on the Wisconsin Breast Cancer dataset to distinguish malignant from benign tumors, with special focus on optimizing recall to minimize false negatives in medical diagnostics.

## Key Features

- Implemented SVM with both linear and RBF kernels
- Visualized decision boundaries using PCA
- Conducted hyperparameter tuning via validation curves
- Performed comprehensive grid search
- Focused on recall metric (critical for medical diagnosis)

## Methodology

### 1. Data Preparation
- Standardized features using `StandardScaler`
- Performed train-test split (80-20 ratio)
- Applied PCA for dimensionality reduction and visualization

### 2. Model Training

```
from sklearn.svm import SVC

# Optimal parameters from validation curves
svm_rbf = SVC(kernel='rbf', C=1.0, gamma=1.0, random_state=42)
svm_rbf.fit(X_train, y_train)
```

### 3. Hyperparameter Tuning
Used validation curves to identify optimal parameters:

![Gamma Validation Curve](Screenshot_2025-06-05_182513.png)
*Validation curve showing optimal gamma = 1.0*

![C Parameter Validation Curve](Screenshot_2025-06-05_182520.png)
*Validation curve showing optimal C = 1.0*

### 4. Decision Boundary Visualization

```
from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(
    svm_rbf,
    X_pca[:, :2],
    response_method="predict",
    cmap=plt.cm.coolwarm,
    alpha=0.8
)
```

![Detailed Decision Boundary](Screenshot_2025-06-05_180329.png)
*Decision boundary on PCA components with support vectors highlighted*

## Results

| Metric          | RBF Kernel | Linear Kernel |
|-----------------|------------|---------------|
| Training Recall | 0.982      | 0.961         |
| Test Recall     | 0.974      | 0.953         |
| Optimal C       | 1.0        | 1.0           |
| Optimal Gamma   | 1.0        | N/A           |



## Dependencies

- Python 3.7+
- scikit-learn >= 1.0
- matplotlib >= 3.5
- numpy >= 1.21
- pandas >= 1.3
- jupyter >= 1.0

## Image Files

All visualization files are included in the repository:
- `Screenshot_2025-06-05_180329.png`: Detailed decision boundary with support vectors
- `Screenshot_2025-06-05_180339.png`: PCA decision boundary overview
- `Screenshot_2025-06-05_182513.png`: Gamma parameter validation curve
- `Screenshot_2025-06-05_182520.png`: C parameter validation curve
```


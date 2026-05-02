# Scikit-Learn Syllabus

**A Machine Learning Library for Training Models**

---

## → Maths Required (Before Phase 0)

**Goal:** Understand the foundational maths behind ML algorithms

**Topics Covered:**

- Basic Statistics (mean, median, mode, variance, standard deviation)
- Probability Concepts (Bayes Theorem, distributions)
- Linear Algebra (vectors, matrices, dot product, matrix multiplication)
- Derivatives and Gradient (chain rule, partial derivatives)
- Distance Metrics (Euclidean, Manhattan, Cosine)
- Concept of Error and Cost Functions
- Logarithms (used in log loss, entropy)

---

## → Phase 0: Foundations of Machine Learning

**Goal:** Understand what Machine Learning is and where it is used

**Topics Covered:**

- What is Machine Learning?
- Difference: AI vs ML vs Deep Learning
- Types of ML: Supervised, Unsupervised, Reinforcement
- Core ML Concepts:
  → Features (X) and Target (y)
  → Model, training, testing, prediction
- Where Scikit-learn fits into the ML workflow
- Why learn Scikit-Learn for real-world projects
- The standard ML pipeline (end-to-end overview)

**Assignment/Project:**

List 3 real-life ML applications and break each into X and y

---

## → Phase 1: Python, NumPy & Pandas for ML

**Goal:** Gain basic programming and data handling skills

**Topics Covered:**

- Python Refresher:
  → Variables, loops, conditions, functions, lists, dictionaries
- NumPy:
  → Creating arrays, indexing, slicing, reshaping
  → Array-level operations
- Pandas:
  → Series vs DataFrame
  → Loading datasets (CSV)
  → Filtering, slicing, subsetting
  → Descriptive statistics: `.head()`, `.info()`, `.describe()`

**Assignment/Project:**

Load a dataset and explore its structure: shape, head, summary, and apply filters

---

## → Phase 1.5: Exploratory Data Analysis (EDA) ⭐ NEW

**Goal:** Understand data visually before building any model

> Real ML work is 60–70% EDA. Never skip this step.

**Topics Covered:**

- Matplotlib Basics:
  → Line plots, bar charts, histograms, scatter plots
- Seaborn for ML-specific visuals:
  → Heatmaps (correlation matrix)
  → Pairplots
  → Boxplots (outlier detection)
  → Distribution plots
- Understanding Skewness and Outliers
- Feature Correlation Analysis
  → Identifying highly correlated features
  → Dropping redundant features
- Asking the right questions from data before modelling

**Assignment/Project:**

Take any dataset (e.g. Titanic, Iris) → create 5 meaningful visualizations → draw 3 insights from the data

---

## → Phase 2: Data Preprocessing

**Goal:** Clean, transform, and prepare data for training

**Topics Covered:**

- Handling Missing Values:
  → `.dropna()`, `.fillna()`, checking with `.isnull()`
  → Mean/Median/Mode imputation strategies
- Encoding Categorical Data:
  → Label Encoding
  → One-Hot Encoding
  → Ordinal Encoding
- Feature Scaling:
  → StandardScaler
  → MinMaxScaler
  → RobustScaler (for data with outliers)
- Splitting Data:
  → `train_test_split(X, y)`
  → Understanding test_size, random_state, shuffle
- Handling Imbalanced Data (intro):
  → Class imbalance problem
  → `class_weight='balanced'`

**Assignment/Project:**

Take a CSV file with missing and categorical data → clean, encode, scale, and split

---

## → Phase 3: Supervised Machine Learning

**Goal:** Train predictive models for classification and regression tasks

**Topics Covered:**

- **Regression:**
  → Linear Regression using `LinearRegression()`
  → Ridge Regression (L2 regularization) ⭐ NEW
  → Lasso Regression (L1 regularization) ⭐ NEW
  → When and why to use regularization

- **Classification:**
  → Logistic Regression
  → K-Nearest Neighbors (KNN)
  → Decision Tree Classifier
  → Random Forest Classifier ⭐ NEW
  → Support Vector Machine (SVM) ⭐ NEW

- Model Training & Prediction:
  → `.fit(X_train, y_train)`
  → `.predict(X_test)`
  → `.predict_proba(X_test)`

- Underfitting vs Overfitting
  → Concepts, visual understanding, and how to fix them
  → Bias-Variance Tradeoff ⭐ NEW

**Assignment/Project:**

1. Predict house prices using size and location features (regression + Ridge/Lasso)
2. Predict student pass/fail using study hours (classification + Random Forest vs Decision Tree)
3. Compare accuracy of all models side-by-side

---

## → Phase 4: Model Evaluation & Metrics

**Goal:** Evaluate model performance using appropriate metrics

**Topics Covered:**

**For Classification:**

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC Score ⭐ NEW
- `classification_report` and `ConfusionMatrixDisplay`

**For Regression:**

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

**Cross-Validation:** ⭐ NEW (moved here from Phase 6)

- `cross_val_score()`
- Why cross-validation gives a more honest performance estimate than a single train/test split
- K-Fold Cross-Validation

**Scikit-learn Modules:**

- `sklearn.metrics`
- `sklearn.model_selection`

**Assignment/Project:**

Use classification and regression models and evaluate them using 3+ metrics each. Also apply 5-Fold Cross-Validation and compare with single split results.

---

## → Phase 5: Unsupervised Learning (Clustering)

**Goal:** Group unlabeled data using similarity-based learning

**Topics Covered:**

- K-Means Clustering:
  → `.fit()`, `.predict()`, `.inertia_`
  → Elbow Method to determine `k`

- DBSCAN Clustering: ⭐ NEW
  → Density-based approach
  → Handles non-circular clusters
  → `eps` and `min_samples` parameters

- Hierarchical Clustering: ⭐ NEW
  → Dendrograms
  → Agglomerative approach

- Principal Component Analysis (PCA):
  → Reducing dimensions
  → `explained_variance_ratio_`, visualizing clusters

- Visualizing clusters using scatter plots
- Cluster Labeling and Interpretability

**Assignment/Project:**

1. Cluster customers by Age and Spending Score using Mall Customer dataset using K-Means
2. Apply DBSCAN and compare cluster shapes with K-Means
3. Visualize clusters and apply PCA to reduce to 2D

---

## → Phase 6: Model Tuning, Pipelines & Deployment

**Goal:** Improve model accuracy, automate workflows, and deploy models

**Topics Covered:**

**Hyperparameter Tuning:**

- `GridSearchCV`
- `RandomizedSearchCV`
- Nested Cross-Validation

**Pipelines:**

- Automate preprocessing + modeling
- `Pipeline()`, `ColumnTransformer()`
- Why pipelines prevent data leakage

**Model Saving & Loading:**

- Using `joblib`
- Using `pickle`

**Deployment (Intro):** ⭐ NEW

- Wrapping a model in a **Flask or FastAPI** REST API endpoint
- Building a simple **Streamlit app** for model demos
  → Input form → prediction output
  → Great for portfolio projects

**Assignment/Project:**

Tune a Random Forest model using GridSearch → build a full Pipeline (preprocessing + model) → save it with joblib → reload it → wrap it in a Streamlit app → demo prediction on unseen data

---

## → Capstone Project ⭐ NEW

**Goal:** Build one complete, end-to-end ML project for your portfolio

**What to Build:**

An end-to-end ML project covering the **entire pipeline**:

1. **Pick a real dataset** (Kaggle, UCI, government data)
2. **EDA** – visualizations, insights, feature correlation
3. **Preprocessing** – missing values, encoding, scaling via Pipeline
4. **Model Selection** – train 3+ models, compare metrics
5. **Hyperparameter Tuning** – GridSearchCV on the best model
6. **Cross-Validation** – verify performance is consistent
7. **Save the Model** – joblib/pickle
8. **Deploy** – Streamlit app or Flask API

**Suggested Project Ideas:**

- Loan Default Prediction (classification)
- House Price Estimator (regression)
- Customer Churn Predictor (classification + imbalanced data)
- Movie/Product Recommendation (unsupervised + collaborative filtering intro)

> This is what goes on your GitHub and resume. Make it count.

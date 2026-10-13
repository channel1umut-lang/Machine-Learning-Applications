# Machine Learning Applications

Hands-on implementations of core supervised learning algorithms with **scikit-learn**, each applied to a realistic dataset, from bank marketing and fraud detection to cyber-attack classification and health-risk prediction. The series ends with a complete, end-to-end ML project.

Every notebook runs in Google Colab: open it and click **"Open in Colab"** at the top.

**Tech:** Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Google Colab

---

## ⭐ Final project: Water Potability Prediction

[`UMUT_YAVAS_ML_FINAL_M505A.ipynb`](UMUT_YAVAS_ML_FINAL_M505A.ipynb)

Can we predict whether water is safe to drink from 9 physical and chemical measurements (pH, hardness, sulfate, turbidity…)? Dataset: [Kaggle Water Quality](https://www.kaggle.com/datasets/adityakadiwal/water-potability), 3,276 samples.

**Pipeline**
1. Train/test split **before** preprocessing to prevent data leakage
2. Median imputation of missing values (fitted on training data only)
3. Feature scaling
4. Four models tuned with 5-fold cross-validation (GridSearchCV / RandomizedSearchCV)
5. Final evaluation of the best model on the held-out test set

**Results**

| Model | CV score |
|---|---|
| **Random Forest** | **0.672** |
| K-Nearest Neighbors | 0.646 |
| Decision Tree | 0.644 |
| Logistic Regression | 0.625 |

Random Forest reached **0.67 test accuracy**, matching its CV score (no overfitting). It identifies unsafe water reliably (94% recall) but misses many potable samples (30% recall), a trade-off discussed in the notebook's conclusion.

---

## 📚 Notebooks

### Regression
| # | Topic | Dataset / use case |
|---|---|---|
| 01 | [Simple Linear Regression](0001_SimpleLinearRegression.ipynb) | Study hours → exam score |
| 02 | [Multiple Linear Regression](0002_MultipleLinearRegression.ipynb) | Student grades |
| 03 | [Polynomial Regression](0003_PolynomalRegression.ipynb) | Customer satisfaction vs. incentives |
| 04 | [Ridge, Lasso & ElasticNet](0004_RidgeLassoElastic.ipynb) | Algerian forest fires |
| 05 | [Comparing Regression Models](0005_RegressionDecision.ipynb) | Diabetes (LazyPredict) |
| 06 | [Scaler Differences](0006_ScalerDifferences.ipynb) | Standard vs. MinMax vs. Robust scaling |

### Classification
| # | Topic | Dataset / use case |
|---|---|---|
| 07 | [Logistic Regression: Intro](0007_LogisticRegressionIntro.ipynb) | Bank term-deposit subscription |
| 08 | [Logistic Regression: Multi-class](0008_LogisticRegressionMultiClass.ipynb) | Cyber-attack type detection |
| 09 | [Logistic Regression: Advanced](0009_LogisticRegressionAdvanced.ipynb) | Fraud detection on imbalanced data |
| 10 | [SVM Classifier](0010_SVMClasiffier.ipynb) | Email classification, loan risk |
| 11 | [SVM Regressor](0011_SVMRegressor.ipynb) | Diamond prices |
| 12 | [Naive Bayes](0012_NaiveBayesClassifier.ipynb) | Iris species |
| 13 | [K-Nearest Neighbors](0013_KNNClassifierAndRegressor.ipynb) | Health risk, house energy use |
| 14 | [Decision Tree](0014_DecisionTreeClassifier.ipynb) | Car evaluation |

### Ensembles
| # | Topic | Dataset / use case |
|---|---|---|
| 15 | [Random Forest Classifier](0015_RandomForestClassifier.ipynb) | Adult income (>50K) |
| 16 | [Random Forest Regressor](0016_RandomForestRegressor.ipynb) | Gym crowdedness |
| 17 | [AdaBoost Classifier](0017_AdaBoostingClassifiar.ipynb) | Pima diabetes |

---

## ▶️ Running locally

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Most notebooks load data from Google Drive (`/content/drive/...`). To run locally, download the dataset and update the `pd.read_csv(...)` path.

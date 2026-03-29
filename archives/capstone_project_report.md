# Capstone Project Report

## Predicting Daily WHOOP Strain Classes from Multimodal Health and Fitness Data

### Project Team

- Vu Hoang Phuc - BIT240184 - Class 24IT-GM - ML Models
- Dang Thai Binh - BIT240042 - Class 24IT-GM - GRU Models
- Tran Vu Luan - BIT240145 - Class 24IT-GM - Evaluation
- Dinh The Duy - BCS240014 - Class 24CS-GM - Data Engineer

### Project Artifacts

- Dataset: `data/whoop_fitness_dataset_100k.csv`
- Experimental notebook: `notebooks/main.ipynb`
- Report date: March 29, 2026

---

## Abstract

Wearable health devices provide rich physiological and behavioral signals that can be leveraged for personalized training decisions. This capstone investigates multi-class prediction of daily training intensity from a large synthetic WHOOP-style dataset containing 100,000 daily records from 286 users over 13 months [1]. The target variable is a 4-level strain class derived from WHOOP strain guidance: Rest/Recovery (0-9), Moderate (10-13), High (14-17), and All Out (18-21) [2].

We designed an end-to-end pipeline including data quality checks, missing value handling, exploratory analysis, feature engineering, leakage prevention, and model benchmarking under a shared protocol. To ensure fair comparison between tabular and sequence models, we applied a user-time consistent split strategy and common test indices. Four models were evaluated: a custom multiclass Logistic Regression, XGBoost [3], a custom GRU, and a Keras GRU reference model [4].

Results show that XGBoost achieved the strongest overall performance with test Accuracy = 0.6519 and Macro-F1 = 0.5077, outperforming Logistic Regression (Macro-F1 = 0.3709) and both GRU variants (Macro-F1 = 0.1794 and 0.2069). Class-wise analysis indicates that gradient boosting preserves useful discrimination for minority high-intensity classes, while sequence models collapsed toward majority-class predictions under current setup.

The main contribution of this project is a reproducible and methodologically controlled benchmark that clarifies which modeling family is most effective for this specific feature-target configuration. We also identify key limitations and practical improvements for future work, including class-imbalance mitigation, richer sequence design, calibrated decision policies, and subject-level generalization tests.

## Keywords

Wearable analytics, WHOOP strain, health informatics, multiclass classification, GRU, XGBoost, imbalanced learning, fitness data mining

---

## 1. Introduction

Digital health tracking has moved beyond simple step counting toward continuous multimodal monitoring of sleep, recovery, cardiovascular response, and exercise load. In this context, WHOOP strain is an actionable indicator of daily physiological stress and training demand [2]. Predicting strain class from known daily signals can support training recommendation systems, overreaching prevention, and adaptive workout planning.

This project addresses the following research questions:

- RQ1: Which model family (traditional machine learning vs. sequence deep learning) is more effective for daily strain class prediction under a fair and controlled protocol?
- RQ2: Can a 7-day temporal context captured by GRU outperform tabular methods on this task?
- RQ3: How does class imbalance influence minority-class detectability, especially for High and All Out workloads?

### 1.1 Project Objectives

- Build a complete and reproducible modeling pipeline from raw data to final comparative evaluation.
- Predict 4-level daily strain class grounded in WHOOP domain interpretation.
- Evaluate models with metrics suitable for imbalanced multiclass classification.
- Provide evidence-based conclusions and practical recommendations.

---

## 2. Dataset and Problem Formulation

## 2.1 Dataset Overview

The dataset contains 100,000 daily records (39 columns) for 286 synthetic users across January 2023 to February 2024 [1]. It combines user profile, sleep, recovery, cardiovascular, and workout variables.

Feature groups include:

- User profile and demographics
- Recovery and daily strain indicators
- Sleep architecture and quality metrics
- Cardiovascular baseline and daily response metrics
- Activity type, duration, strain, calories, and heart-rate zone allocations

The source is synthetic and generated using physiologically informed probabilistic relationships [1]. This design is suitable for education and research while preserving privacy.

> Image Placeholder - Figure 1
> Placement: Directly below Section 2.1 (current position).
> Add: A dataset schema showing feature groups and their column counts.
> Suggested file: `images/fig01_dataset_schema.png`

![Figure 1 Placeholder: Dataset schema and feature groups](images/fig01_dataset_schema.png)
_Figure 1. Dataset schema summarizing user profile, recovery and strain, sleep, cardiovascular, and activity feature groups._

## 2.2 WHOOP Domain Mapping for Target Labels

Following the WHOOP strain interpretation article (February 10, 2026), day strain is mapped to four classes [2]:

- Class 0: Rest/Recovery (strain < 10)
- Class 1: Moderate (10 <= strain < 14)
- Class 2: High (14 <= strain < 18)
- Class 3: All Out (strain >= 18)

This mapping aligns model outputs with actionable training zones and supports recommendation-oriented interpretation.

> Image Placeholder - Figure 2
> Placement: Directly below Section 2.2 (current position).
> Add: A WHOOP strain zone diagram (0-21 scale) with four class boundaries used in this project.
> Suggested file: `images/fig02_strain_zone_mapping.png`

![Figure 2 Placeholder: WHOOP strain class mapping](images/fig02_strain_zone_mapping.png)
_Figure 2. Mapping from day strain score to the four target classes used for model training._

## 2.3 Prediction Task

Given daily physiological and contextual signals available before final day outcomes, predict `strain_class` as a multiclass classification problem.

Important modeling principle: variables that directly reveal same-day workout outcomes are excluded to prevent data leakage.

---

## 3. Methodology

## 3.1 Data Inspection and Cleaning

The notebook pipeline performed the following checks and cleaning actions:

- Initial shape confirmation: (100000, 39)
- Converted `date` to datetime type
- Missing value audit detected 45,990 missing values in `workout_time_of_day`
- Filled missing `workout_time_of_day` with `No Workout`
- Removed `sleep_performance` due to zero variance

These steps established a clean and consistent base for feature engineering and model training.

> Image Placeholder - Figure 3
> Placement: Directly below Section 3.1 (current position).
> Add: Data cleaning summary chart (for example, missing values before and after cleaning).
> Suggested file: `images/fig03_data_cleaning_summary.png`

![Figure 3 Placeholder: Data cleaning summary](images/fig03_data_cleaning_summary.png)
_Figure 3. Data quality summary highlighting missing value handling and final cleaned dataset state._

## 3.2 Exploratory Analysis

Visual diagnostics included:

- Distribution plot of `day_strain`
- Correlation heatmap among key physiological variables
- Daily time trend and lag autocorrelation of mean strain

The EDA confirms that the problem has both multivariate structure and temporal dependence, motivating comparison between tabular and sequential models.

> Image Placeholder - Figure 4
> Placement: Directly below Section 3.2 (current position).
> Add: A 3-panel EDA figure combining strain distribution, correlation heatmap, and temporal trend/autocorrelation.
> Suggested file: `images/fig04_eda_panels.png`

![Figure 4 Placeholder: Combined EDA panels](images/fig04_eda_panels.png)
_Figure 4. Exploratory analysis panels: day strain distribution, physiological correlation matrix, and time-series dependence diagnostics._

## 3.3 Feature Engineering and Leakage Control

Engineered variables:

- `bmi = weight_kg / (height_m^2)`
- `hrv_rhr_ratio = hrv / resting_heart_rate`
- `is_weekend` from date
- `age_group` bins: Young, Adult, Middle-Aged, Senior

Leakage-prone columns were removed, including:

- Target source and direct derivatives (`day_strain`, range-related columns)
- Same-day workout outcomes (`activity_strain`, `activity_calories`, HR zone durations, etc.)
- Metadata not used for learning (`user_id`, `date`)

Final feature set size: 25 variables.

### 3.3.1 Class Distribution

The generated target is imbalanced:

- Class 0 (Rest/Recovery): 53.899%
- Class 1 (Moderate): 27.210%
- Class 2 (High): 13.820%
- Class 3 (All Out): 5.071%

This imbalance directly informed metric selection and interpretation.

## 3.4 Unified Experimental Protocol

To ensure fair model comparison:

- Sequence lookback fixed at 7 days
- Per-user temporal split: 70% train, 15% validation, 15% test
- Shared test index protocol for all models
- Verified no overlap among train/val/test index sets

Observed split sizes:

- Train: 68,472
- Validation: 14,699
- Test: 14,827

Feature preprocessing for tabular models [5]:

- Numeric: StandardScaler
- Categorical: OneHotEncoder (handle_unknown = ignore)

> Image Placeholder - Figure 5
> Placement: Directly below Section 3.4 (current position).
> Add: Experimental protocol diagram from raw data to split, preprocessing, model training, and evaluation.
> Suggested file: `images/fig05_protocol_pipeline.png`

![Figure 5 Placeholder: Experimental protocol and pipeline](images/fig05_protocol_pipeline.png)
_Figure 5. Unified experimental pipeline with user-time split protocol and common evaluation path._

## 3.5 Models

### 3.5.1 Custom Logistic Regression (Multiclass Softmax)

A from-scratch implementation with gradient descent optimization was used as a transparent linear baseline.

### 3.5.2 XGBoost Classifier

XGBoost was used as a strong tree-ensemble reference for nonlinear tabular patterns [3]. Validation Macro-F1 was substantially higher than Logistic Regression.

### 3.5.3 Custom GRU

A manually implemented GRU cell and classifier were trained on 7-day windows with categorical crossentropy and Adam optimizer, following GRU principles in [4].

### 3.5.4 Keras GRU Reference

A two-layer Keras GRU architecture was trained as a framework baseline to compare against the custom GRU implementation [4].

## 3.6 Evaluation Metrics

Given multiclass imbalance, Macro metrics were emphasized:

- Accuracy
- Precision-Macro
- Recall-Macro
- F1-Macro (primary metric)

Macro-F1 is preferred because it gives equal weight to minority classes and better reflects practical fitness recommendation quality.

---

## 4. Experimental Results

## 4.1 Validation-Stage Signals

- Logistic Regression validation Macro-F1: 0.3808
- XGBoost validation Macro-F1: 0.5216

Early validation already suggested XGBoost superiority for this feature regime.

## 4.2 Final Test Results (Common Protocol)

Test set size: 14,827 samples.

| Rank | Model               | Accuracy | Precision-Macro | Recall-Macro | F1-Macro |
| ---- | ------------------- | -------: | --------------: | -----------: | -------: |
| 1    | XGBoost             |   0.6519 |          0.5492 |       0.4862 |   0.5077 |
| 2    | Logistic Regression |   0.6405 |          0.3879 |       0.3805 |   0.3709 |
| 3    | Keras GRU           |   0.5467 |          0.2243 |       0.2592 |   0.2069 |
| 4    | Custom GRU          |   0.5476 |          0.2370 |       0.2509 |   0.1794 |

> Image Placeholder - Figure 6
> Placement: Directly below Section 4.2 result table (current position).
> Add: Comparative bar chart (Accuracy and Macro-F1 by model).
> Suggested file: `images/fig06_model_comparison.png`

![Figure 6 Placeholder: Model performance comparison](images/fig06_model_comparison.png)
_Figure 6. Performance comparison across models, emphasizing Macro-F1 for imbalanced multiclass evaluation._

## 4.3 Class-Wise Behavior

### 4.3.1 Logistic Regression

- Strong on Rest class (F1 = 0.83)
- Moderate class acceptable (F1 = 0.44)
- Weak for High (F1 = 0.22)
- Failed on All Out (F1 = 0.00)

### 4.3.2 XGBoost

- Rest class strong (F1 = 0.82)
- Moderate class stable (F1 = 0.44)
- Improved High class detection (F1 = 0.37)
- Meaningful All Out detection (F1 = 0.39)

### 4.3.3 GRU Models

Both GRU variants underperformed and showed class-collapse tendencies. The custom GRU predicted almost only majority classes on test windows:

- Predicted distribution approximately: Class 0 = 99.66%, Class 1 = 0.34%

This explains very low Macro-F1 despite moderate overall Accuracy.

> Image Placeholder - Figure 7
> Placement: Directly below Section 4.3 (current position).
> Add: Confusion matrix grid for all evaluated models.
> Suggested file: `images/fig07_confusion_matrices.png`

![Figure 7 Placeholder: Confusion matrices by model](images/fig07_confusion_matrices.png)
_Figure 7. Confusion matrices showing class-wise prediction behavior and minority-class errors._

## 4.4 Training Observations

- XGBoost training completed rapidly (about 1.04 seconds in recorded run)
- GRU training converged to validation accuracy near majority-class prevalence, signaling poor minority discrimination
- Early stopping and learning-rate scheduling did not resolve class-collapse under current setup

---

## 5. Discussion

## 5.1 Why XGBoost Won

The feature set is dominated by strong daily tabular predictors (recovery, sleep, HRV, RHR context, demographics, encoded behavior categories). Gradient boosting effectively captures nonlinear interactions among these features without requiring long temporal memory. Under this data representation, tabular boosting is more suitable than sequence RNNs.

## 5.2 Why GRU Underperformed

Several factors likely contributed:

- Severe class imbalance with low prevalence of classes 2 and 3
- Sequence length (7 days) may be insufficient to capture robust temporal signatures for rare high-strain events
- Categorical variables were label-encoded for GRU, potentially introducing artificial ordinal relations
- No class weighting, focal loss, or imbalance-aware sampling in GRU training

Together, these factors encouraged majority-class dominance.

## 5.3 Practical Implications

From a coaching decision perspective, minority-class sensitivity matters because High and All Out states often trigger workload caution or adaptive planning. A model with high Accuracy but weak minority recall is less useful for safety and performance management.

## 5.4 Alignment with WHOOP Strain Logic

The label design is consistent with WHOOP guidance [2] and supports an operational decision framework:

- Class 0: prioritize recovery or very light activity
- Class 1: moderate maintenance load
- Class 2: higher training stimulus when recovery supports it
- Class 3: overreaching-risk zone requiring careful recovery management

Using this taxonomy makes outputs interpretable for autoregulated training strategies.

> Image Placeholder - Figure 8
> Placement: Directly below Section 5.4 (current position).
> Add: Practical decision chart linking predicted class to training recommendation intensity.
> Suggested file: `images/fig08_decision_guideline.png`

![Figure 8 Placeholder: Training decision guideline](images/fig08_decision_guideline.png)
_Figure 8. Suggested decision-support mapping from predicted strain class to actionable training guidance._

---

## 6. Limitations and Future Work

## 6.1 Limitations

- Data is synthetic; external validity to real populations requires caution
- Same-user temporal split evaluates future prediction for known users, not cold-start generalization to unseen users
- Hyperparameter search was limited
- Sequence branch lacked explicit imbalance countermeasures

## 6.2 Recommended Next Steps

- Add class weights or focal loss for GRU
- Use sequence architectures with richer handling of categorical context (embeddings, temporal attention)
- Evaluate LightGBM/CatBoost and calibrated probability thresholds
- Conduct user-level holdout experiments (train on subset of users, test on unseen users)
- Add model explanation (e.g., SHAP) to support feature-level interpretability
- Build two-stage pipelines (first detect workout intensity regime, then classify strain zone)

---

## 7. Team Contributions

- Vu Hoang Phuc (BIT240184, 24IT-GM): Traditional ML modeling and comparative baselines
- Dang Thai Binh (BIT240042, 24IT-GM): GRU sequence modeling (custom and Keras reference)
- Tran Vu Luan (BIT240145, 24IT-GM): Evaluation framework, metric analysis, model comparison
- Dinh The Duy (BCS240014, 24CS-GM): Data engineering, preprocessing, pipeline structuring

---

## 8. Conclusion

This capstone established a complete benchmark for multiclass WHOOP strain prediction using a large multimodal fitness dataset. Under a unified and fair protocol, XGBoost delivered the best balance between overall accuracy and minority-class sensitivity, achieving the top Macro-F1. Linear Logistic Regression remained competitive as a lightweight baseline, while GRU-based sequence models underperformed due to imbalance and representation challenges.

The study demonstrates an important practical point in health analytics: model family selection must be aligned with feature characteristics and decision priorities, not only with temporal model popularity. For this dataset and formulation, robust tabular boosting is currently the most suitable approach. Future work should focus on imbalance-aware deep sequence learning and user-generalization evaluation to improve deployment readiness.

---

## References

[1] L. Gedipudi, "Whoop_Fitness_Dataset [100K]," Kaggle.com, 2023. [Online]. Available: https://www.kaggle.com/datasets/likithagedipudi/whoop-fitness-dataset

[2] WHOOP, "WHOOP Strain Explained: How Your Effort Is Measured," Feb. 10, 2026.

[3] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.

[4] K. Cho et al., "Learning Phrase Representations using RNN Encoder Decoder for Statistical Machine Translation," arXiv preprint arXiv:1406.1078, 2014.

[5] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, 2011.

# Patient 30-Day Readmission Prediction

## Project Overview

This project develops a machine learning model to predict whether a patient is likely to be readmitted to the hospital within 30 days of discharge.

The project uses a synthetic dataset of 5,000 patient records and follows an end-to-end machine learning workflow, including data understanding, exploratory data analysis, preprocessing, model training, evaluation, interpretation, and prediction on new patient data.

The prediction point was defined as **the time of patient discharge**. Therefore, only information available at or before discharge was used as model input. Variables that would only become available after discharge were excluded to avoid data leakage.

Two classification approaches were evaluated: Logistic Regression and Random Forest. The final leakage-safe Logistic Regression pipeline achieved a recall of **61.9%** and a ROC-AUC of **0.695** on the held-out test set.

> **Important:** This project uses synthetic data for educational and demonstration purposes. The model has not been clinically validated and should not be used for real-world clinical decision-making. 
## Dataset

The dataset contains **5,000 synthetic patient records** created for this machine learning demonstration.

Each record contains demographic, healthcare utilization, clinical, and discharge-related information.

### Target Variable

The target variable is:

- `Readmitted_30_Days`
  - `0` = Not readmitted
  - `1` = Readmitted

### Features Used

The model uses the following information available at the point of discharge:

- Age
- Sex
- Length of Stay
- Previous Admissions
- Emergency Visits
- Chronic Conditions
- Number of Medications
- Abnormal Lab Count
- Previous Readmission
- Insurance Type
- Discharge Disposition

### Features Excluded

- `Patient_ID` — excluded because it is an identifier rather than a predictive feature.
- `Follow_Up_Completed` — excluded because it represents information occurring after discharge and could introduce data leakage.
- `Readmitted_30_Days` — excluded from the input features because it is the target being predicted.

> **Data note:** The dataset is synthetic and does not contain real patient records. The simulated relationships and readmission distribution should not be interpreted as representative of real-world clinical populations .
## Exploratory Data Analysis

The exploratory analysis was performed to understand the distribution of the target variable and examine relationships between patient characteristics and 30-day readmission.

### Key Findings

- The target variable was imbalanced, with approximately **81% of patients not readmitted** and **19% readmitted**.
- **Previous admissions** generally showed higher readmission rates as the number of previous admissions increased.
- **Emergency visits** generally showed higher readmission rates at higher visit counts, although very small groups showed more variation.
- **Chronic conditions** showed a general positive relationship with readmission.
- **Previous readmission** was associated with higher observed readmission rates.
- **Length of stay** showed a generally increasing pattern in readmission rates, although the relationship was not perfectly consistent.
- **Age** did not show a strong relationship with readmission.
- **Number of medications** did not show a clear monotonic relationship.
- **Insurance type** showed relatively small differences in readmission rates.
- **Sex** showed a small difference in observed readmission rates.
- **Discharge disposition** showed modest differences between categories.

Because the target variable was imbalanced, accuracy alone was not considered sufficient for evaluating model performance. Metrics such as precision, recall, F1 score, and ROC-AUC were therefore used.

## Data Preprocessing

The dataset was split into training and testing sets using an **80/20 split** with stratification to preserve the proportion of readmitted and non-readmitted patients.

### Preprocessing Steps

- Numerical features were standardized using `StandardScaler`.
- Categorical features were transformed using `OneHotEncoder`.
- The preprocessing steps were combined using a `ColumnTransformer`.
- The preprocessing and Logistic Regression model were combined into a single `Pipeline`.

Using a pipeline ensured that preprocessing was consistently applied during both model training and prediction.

### Data Leakage Prevention

The prediction point was defined as **patient discharge**.

`Follow_Up_Completed` was excluded because follow-up completion occurs after discharge and would not be available at the time the prediction is made.

The final pipeline therefore uses only information available at the defined prediction poin

## Model Development

Two classification algorithms were evaluated:

### Logistic Regression

Logistic Regression was used as the baseline model. Because the target variable was imbalanced, `class_weight="balanced"` was used to give greater importance to the minority readmission class.

### Random Forest

A Random Forest classifier was also evaluated to determine whether a nonlinear ensemble model could improve predictive performance.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 66.3% | 30.6% | 61.9% | 41.0% | 0.695 |
| Random Forest | 80.9% | 40.0% | 2.1% | 4.0% | 0.662 |

Although Random Forest achieved higher accuracy and precision at the default threshold, it had substantially lower recall and missed most of the patients who were actually readmitted.

The balanced Logistic Regression model was therefore selected as the final model because identifying the minority readmission class was a key objective of the project.

## Final Model Evaluation

The final leakage-safe Logistic Regression pipeline was evaluated on a held-out test set of 1,000 patients.

| Metric | Result |
|---|---:|
| Accuracy | 66.3% |
| Precision | 30.6% |
| Recall | 61.9% |
| F1 Score | 41.0% |
| ROC-AUC | 0.695 |

The confusion matrix showed:

- **True Negatives:** 546
- **False Positives:** 265
- **False Negatives:** 72
- **True Positives:** 117

The model correctly identified 117 of the 189 patients who were actually readmitted, resulting in a recall of approximately 61.9%.

The model also produced 265 false positives. This indicates that while the model was able to identify a meaningful proportion of actual readmissions, it also generated a considerable number of false alarms.

The ROC-AUC of 0.695 indicates modest/fair discriminatory ability, meaning that the model performed better than random classification but did not demonstrate strong separation between the two classes.

### Model Interpretation

The strongest positive model coefficients were associated with:

1. Chronic Conditions
2. Previous Readmission
3. Previous Admissions
4. Emergency Visits
5. Length of Stay

These patterns were broadly consistent with findings from the exploratory analysis.

The coefficients represent associations learned by the model and should not be interpreted as causal effects. Because the dataset is synthetic, these findings should not be interpreted as clinical risk factors.

## Example Prediction

The saved model pipeline was tested on a hypothetical new patient using information available at discharge.

The patient had the following characteristics:

- Age: 65
- Sex: Male
- Length of Stay: 8 days
- Previous Admissions: 3
- Emergency Visits: 2
- Chronic Conditions: 4
- Number of Medications: 7
- Abnormal Lab Count: 2
- Previous Readmission: Yes
- Insurance Type: Public
- Discharge Disposition: Home

The model predicted:

- **Predicted class:** Readmitted within 30 days (`1`)
- **Estimated probability of the readmission class:** 81.97%
- **Estimated probability of the non-readmission class:** 18.03%

This demonstrates that the saved pipeline can accept raw patient features, automatically apply the required preprocessing, and generate a prediction.

> **Important:** The probability is an output of a model trained on synthetic data and should not be interpreted as a validated clinical risk estimate.
## Limitations and Ethical Considerations

This project has several important limitations:

- **Synthetic data:** The dataset contains simulated patient records and does not represent real-world patient populations or hospital data.
- **No clinical validation:** The model has not been validated using real clinical data or in a clinical setting.
- **Artificial relationships:** The relationships between variables and the target were intentionally designed during data generation and may not reflect real clinical relationships.
- **Class imbalance:** Approximately 19% of observations belong to the readmission class, making accuracy alone insufficient for evaluating performance.
- **False positives and false negatives:** The final model produced both types of errors. In a real healthcare setting, the consequences of these errors would require careful consideration.
- **Probability interpretation:** Model probabilities should not be treated as validated clinical risk estimates.
- **Correlation is not causation:** Associations identified by the model do not establish that any feature causes hospital readmission.
- **Responsible use:** A real-world implementation would require patient privacy protections, security, governance, clinical oversight, and validation before use.

This project is intended as an educational demonstration of an end-to-end machine learning workflow and should not be used to make clinical decisions.

## Technologies Used

- **Python**
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical computing and synthetic data generation
- **Matplotlib** — data visualization
- **Scikit-learn** — preprocessing, model development, and evaluation
- **Jupyter Notebook** — exploratory analysis and model development
- **VS Code** — project development and data generation
- **Git & GitHub** — version control and project hosti ## Project Structure

## Project Structure

```text
Patient-Readmission-Prediction/
│
├── Data/
│   └── raw/
│       └── patient_readmission_raw.csv
│
├── Models/
│   └── final_logistic_pipeline.joblib
│
├── Notebooks/
│   ├── 01_data_understanding_and_eda.ipynb
│   └── 02_model_testing.ipynb
│
├── SRC/
│   └── data_generation.py
│
├── .gitignore
└── README.m
\#### 1. Clone the Repository

```bash
git clone https://github.com/dmwaura-arch/patient-readmission-prediction.git
cd patient-readmission-predictio```n
``### 2. Install the Required Libraries``bash
pip install pandas numpy matplotlib scikit-learn joblib jupyter
```

### 3. Generate the Synthetic Dataset

Run the data generation script:

```bash
python SRC/data_generation.py
```

This creates the synthetic dataset in:

```text
Data/raw/patient_readmission_raw.csv
```

### 4. Open the Jupyter Notebooks

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open the notebooks in the `Notebooks/` folder.

### 5. Model Testing

The trained model is saved in:

```text
Models/final_logistic_pipeline.joblib
```

The second notebook demonstrates how the saved pipeline can be loaded and used to make predictions on new patient data.

> **Note:** This project is intended for educational and demonstration purposes only.
## Conclusion

This project demonstrates an end-to-end machine learning workflow for a healthcare prediction problem, from synthetic data generation and exploratory analysis to preprocessing, model development, evaluation, interpretation, and testing on new patient data.

The results highlight the importance of selecting evaluation metrics that align with the problem. In this case, recall was particularly important because missing patients who may be readmitted can be more concerning than generating additional false positives.

The final Logistic Regression pipeline achieved a recall of **61.9%** and a ROC-AUC of **0.695** on the held-out test set.

Although the model demonstrated useful predictive patterns within the synthetic dataset, it is not clinically validated and should not be used for real-world clinical decision-making.

This project strengthened my understanding of applying machine learning concepts to a healthcare context while considering data leakage, class imbalance, model evaluation, interpretability, and responsible use of predictive models. 
# MachineGuard: Equipment Failure Prediction Using Machine Learning

## Project Overview

MachineGuard is a machine learning project that explores the application of predictive maintenance techniques to industrial equipment.

The project uses operational measurements to identify machine failure conditions, with an emphasis on minimizing missed failures while reducing unnecessary maintenance alerts.

## Problem Definition

Can a machine learning model accurately classify equipment failure conditions using temperature, rotational speed, torque, and tool wear measurements?

### Project Objectives

- Achieve a minimum recall of 80%.
- Achieve a minimum precision of 50%.
- Investigate the impact of domain-informed feature engineering.
- Optimize the balance between failure detection and false alarms.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Joblib

## Dataset

This project uses the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository.

- Total observations: 10,000
- Original input features: 6
- Machine failures: 339
- Non-failure observations: 9,661

The dataset presents an imbalanced binary classification problem, making accuracy alone an insufficient evaluation metric.

[Dataset Source — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

## Machine Learning Workflow

1. Problem definition
2. Data exploration and preprocessing
3. Stratified training, validation, and test splitting
4. Class imbalance management
5. Baseline model development
6. Comparative model evaluation
7. Domain-informed feature engineering
8. Classification threshold optimization
9. Independent test evaluation
10. Model serialization

## Exploratory Data Analysis

Exploratory data analysis revealed several notable patterns:

- Failure observations generally exhibited higher torque.
- Failed machines tended to have greater accumulated tool wear.
- Torque and rotational speed demonstrated a strong inverse relationship.
- Failure observations appeared at multiple operating extremes, suggesting nonlinear relationships between operational features.

These observations motivated the exploration of feature interactions and ensemble learning.

## Feature Engineering

Three additional features were introduced using engineering principles.

| Feature | Calculation |
|---|---|
| Mechanical Power | Torque × Angular Velocity |
| Temperature Difference | Process Temperature − Air Temperature |
| Mechanical Strain Indicator | Torque × Tool Wear |

The mechanical strain indicator represents a constructed interaction feature rather than a direct physical measurement of strain.

## Model Experimentation

Two classification algorithms were investigated: Logistic Regression and Random Forest.

### Validation Results

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Logistic Regression | 15.08% | 79.41% | 0.2535 |
| Random Forest | 88.89% | 35.29% | 0.5053 |
| Random Forest + Feature Engineering | 95.92% | 69.12% | 0.8034 |
| Random Forest + Feature Engineering + Threshold Optimization | 91.67% | 80.88% | 0.8594 |

The combination of domain-informed feature engineering and classification threshold optimization substantially improved model performance.

## Final Model Performance

The optimized Random Forest classifier was evaluated on an independent test dataset containing 2,000 observations.

| Metric | Result |
|---|---:|
| Accuracy | 99.10% |
| Precision | 89.06% |
| Recall | 83.82% |
| F1-score | 0.8636 |
| Average Precision | 0.8647 |
| Classification Threshold | 0.30 |

### Confusion Matrix Results

- True Positives: 57
- True Negatives: 1,925
- False Positives: 7
- False Negatives: 11

The final model successfully achieved both initial project objectives on unseen test data.

## Installation

Clone the repository:

```bash
git clone https://github.com/chibzdee/MachineGuard.git
cd MachineGuard
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the MachineGuard notebook to explore the complete machine learning workflow.

## Limitations

- The dataset consists of synthetically generated industrial operating measurements.
- The model classifies observed failure conditions rather than forecasting future failures.
- Some engineered features reflect the dataset's failure-generation mechanisms.
- Real-world industrial validation is required before operational deployment.

## Future Improvements

- Integration with real-time industrial sensor data
- Time-series forecasting
- Automated equipment monitoring
- Maintenance alert systems
- Model deployment through a REST API
- Validation using real-world industrial equipment data

## Author

**Chibuikem Didigu**

Computer Science Student | Machine Learning & AI Engineering Enthusiast

Developed as an independent project to explore the applications of machine learning in industrial equipment reliability and predictive maintenance.

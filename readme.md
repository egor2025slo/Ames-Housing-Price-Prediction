# Ames Housing Price Prediction

## Project Overview
This project focuses on building a machine learning pipeline to predict house sale prices using the Ames Housing dataset. The goal is to apply end-to-end data science practices including data cleaning, feature engineering, preprocessing pipelines, model training, evaluation, and interpretation.

## Business Problem
Accurate house price prediction is important for real estate agencies, investors, and buyers to estimate property value based on structural characteristics, location, and condition.

The task is formulated as a regression problem where the target variable is the sale price of a house.

## Dataset
The Ames Housing dataset contains information about 2,930 residential properties with numerical and categorical features describing:

- Property size and layout
- Construction quality and condition
- Year built and remodeling history
- Garage, basement, and amenities
- Zoning and neighborhood information

Target variable:
- `SalePrice`

## Approach

### 1. Data Exploration (EDA)
- Distribution analysis of house prices
- Correlation analysis for numerical features
- Price comparison across categorical variables
- Outlier detection

### 2. Data Cleaning
- Handling missing values using median and most frequent strategies
- Correcting data types
- Outlier treatment

### 3. Feature Engineering
Key engineered features include:

- `total_house_area` = above ground living area + basement area  
- `total_finished_area`
- `house_age` = year sold - year built
- `remod_age` = year sold - year remodeled
- `bathrooms_total`
- `quality_area` = house area × overall quality score

### 4. Preprocessing Pipeline
Implemented using `scikit-learn`:

- ColumnTransformer for numeric and categorical features
- Missing value imputation
- One-hot encoding for categorical variables
- Feature scaling for linear models

### 5. Models
Two models were trained and compared:

- Ridge Regression (baseline)
- Gradient Boosting Regressor (strong model)

Hyperparameters were tuned using cross-validation.

---

## Results

### Cross-Validation Performance (RMSE in log-space)

- Ridge Regression: 0.135
- Gradient Boosting: 0.123

### Test Set Performance

#### Log-scale metrics:

| Model | MAE | RMSE | R² |
|------|------|------|------|
| Ridge | 0.078 | 0.113 | 0.932 |
| Gradient Boosting | 0.072 | 0.105 | 0.941 |

#### Original price scale:

| Model | MAE ($) | RMSE ($) | R² |
|------|---------|----------|------|
| Ridge | 15,432 | 28,641 | 0.898 |
| Gradient Boosting | 13,659 | 23,318 | 0.932 |

The Gradient Boosting model achieved the best performance with an average prediction error of approximately $13,600.

---

## Model Diagnostics

- Predicted vs actual price scatter plots showed strong alignment
- Residuals were centered around zero with no strong patterns
- Error analysis by price segment showed slightly higher errors for cheaper houses

---

## Feature Importance (Gradient Boosting)

Top contributing features:

- quality_area
- Overall Qual
- total_house_area
- total_finished_area
- house_age
- Year Built
- Garage Cars
- Central Air
- Lot Area

These features align well with real-world housing valuation logic.

---

## Limitations

- The dataset is limited to a specific region (Ames, Iowa)
- Market trends over time were not explicitly modeled
- Some engineered features may be correlated

---

## Future Work

- Apply more advanced boosting models (LightGBM, XGBoost)
- Use SHAP values for deeper model interpretation
- Add neighborhood-level aggregated features
- Experiment with quantile regression for uncertainty estimation

---

## Technologies Used

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

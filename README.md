# Real Estate Price Prediction Model

This project implements a machine learning model to predict real estate prices based on various features, such as location, property characteristics, and economic indicators. The model uses the CatBoost algorithm, which is tuned for optimal performance.

## Overview

The goal of this project is to predict the price of properties using various features that represent property attributes, location, and related economic factors. The model is evaluated using multiple regression metrics and feature importance analysis.


## Project Structure
```plaintext
IMMOELIZA-ML/
├── Data/
│   ├── immoweb_data_cleaned.csv          # Cleaned dataset
│   ├── immoweb_features2.csv             # Feature-engineered dataset
│   ├── immoweb_with_all_columns.csv      # Dataset with all original columns
├── Features/                             # Directory containing all scripts and data related to feature engineering, including scripts for feature creation and external data files
├── Figures/                              # Folder for additional plots and visualizations
│   ├── Feature Importances.png           # Visualization of feature importance
│   ├── Residual Distribution.png         # Residual distribution plot
│   ├── Residuals vs Predicted Values.png # Predicted vs residual values
│   ├── SHAP bar plot.png                 # SHAP bar plot for feature importance
│   └── SHAP summary plot.png             # SHAP summary plot                   
├── catboost_info/                        # Folder containing CatBoost training logs
├── Data_cleaning.py                      # Script for cleaning data
├── Feature_Engineering.py                # Script for feature engineering
├── Housing_Price_Prediction_Report.ipynb # Jupyter Notebook with project details
├── Model_Training.py                     # Main script for model training
├── main.py                               # Entry point script (optional)
├── plot.py                               # Script for generating plots
└── predictions.csv                       # Final model predictions(Actual Prices,Predicted Prices)
└── requirements.txt                      # Python dependencies
└── TEST.py                               #script serves as a utility to test the trained model using a randomly generated dataset.
```

## Installation

To run this project locally, follow the steps below:

1. Clone the repository:
    ```bash
    git clone https://github.com/ManelBouba/ImmoEliza-ML.git
    cd ImmoEliza-ML
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure that you have all necessary files, including the dataset (`data.csv`) and the trained model (`catboost_model_with_tuning.cbm`).
4. Exponentiate the predictions to get the original price scale predicted_prices = np.exp(predictions)

## Dataset

The dataset used in this project contains information about properties and various features, such as:
- `Province`, `Locality`, `Municipality`
- `Living_Area`, `Living_Area_log`
- `State_of_the_Building`, `Subtype_of_Property`
- Economic indicators: `Unemployment Rate (%)`, `Population Density`, etc.

These features are used to predict the `Property_Price`.

## Model

The model is trained using the CatBoost algorithm, a gradient boosting technique known for its high performance with categorical data. The model is fine-tuned using hyperparameter optimization to ensure optimal prediction results.

### Training and Testing Results

The model has been evaluated using the following metrics:

- **Training RMSE**: 60,953.93
- **Test RMSE**: 66,504.85
- **Training MAE**: 44,866.97
- **Test MAE**: 48,726.73
- **Training R²**: 0.784
- **Test R²**: 0.748
- **Training MAPE**: 14.37%
- **Test MAPE**: 15.78%

### Residuals Analysis:
- **Mean of residuals**: 4,367.32
- **Standard deviation of residuals**: 66,361.30
- **Max residual**: 363,595.28
- **Min residual**: -246,333.91

These statistics indicate a reasonable model fit, with some outliers that might be worth investigating further.

### Feature Importance:
The following features were found to be most impactful in the model's predictions:
1. `Province` (16.4%)
2. `Living_Area` (11.88%)
3. `Living_Area_log` (11.46%)
4. `State_of_the_Building` (9.74%)
5. `Locality` (8.66%)

These features have the highest importance in determining property prices.

### SHAP Analysis:
SHAP (SHapley Additive exPlanations) values were used to interpret the model's predictions. Visualizations show how each feature affects individual predictions, making the model more interpretable.

## How to Use

1. **Loading the Model**: The trained CatBoost model can be loaded from the saved file:
    ```python
    import catboost
    model = catboost.CatBoostRegressor()
    model.load_model("catboost_model_with_tuning.cbm")
    ```

2. **Making Predictions**: To make predictions on new data, prepare the data as a DataFrame and use the model's `predict` method:
    ```python
    import pandas as pd
    data = pd.read_csv("new_data.csv")
    predictions = model.predict(data)
    ```

3. **Output**: The predictions will be saved in `predictions.csv` by default.

## Visualizations

- **Feature Importance Plot**: A bar plot visualizes the importance of each feature in the model.
- **SHAP Summary Plot**: A plot that shows the global importance of features.
- **SHAP Bar Plot**: A bar chart summarizing the average impact of each feature on the model’s predictions.
- **SHAP Single Prediction Visualization**: Visualizes the impact of each feature on a single prediction.

## Future Improvements

- **Outlier Handling**: Further work can be done to handle large residuals and outliers. Techniques like robust regression or outlier removal could be explored.
- **Cross-Validation**: Implement cross-validation to get a more accurate estimate of the model’s performance on unseen data.
- **Hyperparameter Tuning**: More extensive hyperparameter tuning might lead to a further improvement in model performance.
- **Feature Engineering**: Additional feature engineering, such as creating interaction terms, might help improve the model.

## Conclusion

This project demonstrates the power of machine learning in predicting property prices using various features. The CatBoost model performs well with a solid R² value of around 0.75. By improving feature engineering and handling outliers, future iterations of the model can achieve even better results.

# References

This project utilizes several sources for data and tools, which are crucial for building and training the CatBoost model for real estate price prediction in Belgium.

## 1. **Data Sources**  
The following datasets and external data sources were used in this project:

- **Immoweb Data**: The primary dataset was extracted from [Immoweb](https://www.immoweb.be/en), one of Belgium's leading real estate websites. This dataset includes property prices and various property features.
- **External Economic and Demographic Data**: Additional features were created using external datasets, including economic indicators like unemployment rates, population density, and geographic information. These datasets were sourced from:
  - [Statistics Belgium](https://statbel.fgov.be/en) for economic and population statistics.
  - [Open Data Portal - Brussels](https://data.brussels.be/en) for geographic and regional data.
  - [European Union Open Data Portal](https://data.europa.eu/euodp/en/home) for broader European economic indicators, including unemployment rates and housing data.
  
## 2. **Feature Engineering**  
External data such as population density, unemployment rates, and geographical data were incorporated into the model to enhance its prediction accuracy. The following resources and techniques were employed in the feature engineering process:
  - [Scikit-learn](https://scikit-learn.org/stable/) for creating machine learning pipelines, transformations, and model evaluation.
  - [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap) for model explainability and feature importance visualization.
  - [Pandas](https://pandas.pydata.org/) for data manipulation and cleaning.
  
## 3. **Modeling and Tools Used**
  - **CatBoost**: The machine learning model used in this project is [CatBoost](https://catboost.ai/), a gradient boosting library by Yandex, which is particularly effective with categorical features. The version used in this project is `catboost==1.2`.
  - **Matplotlib** and **Seaborn** for plotting and data visualization:
    - [Matplotlib](https://matplotlib.org/) for general plotting.
    - [Seaborn](https://seaborn.pydata.org/) for more advanced statistical visualizations, including distribution and correlation plots.

## 4. **Model Validation and Performance Evaluation**  
The model was evaluated using common regression metrics such as RMSE, MAE, and R², and the results were visualized using:
  - [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) for computing evaluation metrics.
  - [SHAP](https://github.com/slundberg/shap) for understanding model predictions and feature contributions.

## 5. **External Libraries and Dependencies**  
For a complete list of dependencies used in this project, see the [requirements.txt](requirements.txt) file, which includes libraries such as Pandas, Numpy, Scikit-learn, Matplotlib, and SHAP.

## 6. **References for Data Processing and Feature Engineering**  
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Feature Engineering - Kaggle](https://www.kaggle.com/learn/feature-engineering)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
  


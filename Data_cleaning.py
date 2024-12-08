# data_cleaning.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class DataCleaning:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.categorical_features = [
            'Locality', 'Type_of_Property', 'Subtype_of_Property', 
            'State_of_the_Building', 'Fully_Equipped_Kitchen', 
            'Terrace', 'Garden', 'Swimming_Pool', 'Lift', 
            'Municipality', 'Province'
        ]

    def clean_data(self):
        """Handle missing values, outliers, and feature transformations."""
        # Handle missing values in categorical features
        for col in self.categorical_features:
            self.df[col] = self.df[col].fillna('Unknown')
            self.df[col] = self.df[col].apply(lambda x: str(x) if isinstance(x, (float, int)) else x)

        # Handle outliers using Isolation Forest
        numerical_features = self.df.select_dtypes(include=[np.number]).columns
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso_forest.fit_predict(self.df[numerical_features])

        # Keep only the inliers (outliers = -1, inliers = 1)
        self.df = self.df[outliers == 1]

        # Apply log transformation to the 'Price' column
        self.df['Log_Price'] = np.log1p(self.df['Price'])

        return self.df

    def get_features_and_target(self):
        """Return the features and target variable for modeling."""
        X = self.df.drop(columns=['Price', 'Log_Price'])
        y = self.df['Log_Price']
        return X, y

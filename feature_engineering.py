import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, df):
        """
        Initializes the feature engineering class with the provided DataFrame.
        """
        self.df = df
    
    def add_price_per_m2(self):
        """
        Adds a feature for the price per square meter (Price / Living_Area).
        """
        if 'Price' in self.df.columns and 'Living_Area' in self.df.columns:
            self.df['price_per_m2'] = self.df['Price'] / self.df['Living_Area']
        else:
            print("Error: 'Price' or 'Living_Area' column is missing.")
    
    def add_price_per_room(self):
        """
        Adds a feature for the price per room (Price / Number_of_Rooms).
        """
        if 'Price' in self.df.columns and 'Number_of_Rooms' in self.df.columns:
            self.df['price_per_room'] = self.df['Price'] / self.df['Number_of_Rooms']
        else:
            print("Error: 'Price' or 'Number_of_Rooms' column is missing.")
    
    def add_price_per_facade(self):
        """
        Adds a feature for the price per facade (Price / Number_of_Facades).
        """
        if 'Price' in self.df.columns and 'Number_of_Facades' in self.df.columns:
            self.df['price_per_facade'] = self.df['Price'] / self.df['Number_of_Facades']
        else:
            print("Error: 'Price' or 'Number_of_Facades' column is missing.")
    
    def add_amenities_count(self):
        """
        Adds a feature that counts the number of amenities (e.g., Fully_Equipped_Kitchen, Terrace, etc.).
        """
        amenities_columns = ['Fully_Equipped_Kitchen', 'Terrace', 'Garden', 'Swimming_Pool', 'Lift']
        self.df['amenities_count'] = self.df[amenities_columns].sum(axis=1)

    def add_interaction_features(self):
        """
        Adds interaction features between existing features (e.g., 'Living_Area' and 'Number_of_Rooms').
        """
        if 'Living_Area' in self.df.columns and 'Number_of_Rooms' in self.df.columns:
            self.df['Living_Area_Rooms_interaction'] = self.df['Living_Area'] * self.df['Number_of_Rooms']
    
    def add_aggregated_features(self):
        """
        Adds aggregated features for categorical columns (e.g., average 'Price' per 'Locality').
        """
        if 'Locality' in self.df.columns:
            locality_avg_price = self.df.groupby('Locality')['Price'].transform('mean')
            self.df['avg_price_per_locality'] = locality_avg_price
        
        if 'Province' in self.df.columns:
            province_median_area = self.df.groupby('Province')['Living_Area'].transform('median')
            self.df['median_Living_Area_per_province'] = province_median_area

    def apply_feature_transformations(self):
        """
        Applies transformations (e.g., logarithmic or square root) on highly skewed numerical features.
        """
        numerical_features = ['Price', 'Living_Area']
        
        # Apply logarithmic transformation to 'Price' and 'Living_Area' if they are skewed
        for feature in numerical_features:
            if feature in self.df.columns:
                self.df[feature + '_log'] = np.log1p(self.df[feature])  # log1p handles zero values

    def handle_missing_data(self):
        """
        Handles missing data by imputing with the median for numerical features and mode for categorical ones.
        """
        # Impute missing numerical features with median
        numerical_features = self.df.select_dtypes(include=[np.number]).columns
        for feature in numerical_features:
            if self.df[feature].isnull().sum() > 0:
                self.df[feature].fillna(self.df[feature].median(), inplace=True)
        
        # Impute missing categorical features with mode
        categorical_features = self.df.select_dtypes(include=[object]).columns
        for feature in categorical_features:
            if self.df[feature].isnull().sum() > 0:
                self.df[feature].fillna(self.df[feature].mode()[0], inplace=True)

    def perform_feature_engineering(self):
        """
        Runs all feature engineering methods to enhance the dataset.
        """
        self.add_price_per_m2()
        self.add_price_per_room()
        self.add_price_per_facade()
        self.add_amenities_count()
        self.add_interaction_features()
        self.add_aggregated_features()
        self.apply_feature_transformations()
        self.handle_missing_data()
        
        print("Feature engineering completed.")
        return self.df
    
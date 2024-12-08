# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class FeatureEngineering:
    def __init__(self, df):
        """Initializes the feature engineering class with the provided DataFrame."""
        self.df = df

    def handle_missing_data(self):
        """
        Handles missing data by imputing:
        - Numerical features: median
        - Categorical features: mode
        """
        numerical_features = self.df.select_dtypes(include=[np.number]).columns
        for feature in numerical_features:
            if self.df[feature].isnull().sum() > 0:
                self.df[feature].fillna(self.df[feature].median(), inplace=True)
        
        categorical_features = self.df.select_dtypes(include=[object]).columns
        for feature in categorical_features:
            if self.df[feature].isnull().sum() > 0:
                self.df[feature].fillna(self.df[feature].mode()[0], inplace=True)

    def add_total_area(self):
        """Adds a feature for the total area (Living_Area + Surface_area_plot_of_land)."""
        if 'Living_Area' in self.df.columns and 'Surface_area_plot_of_land' in self.df.columns:
            self.df['Total_Area'] = self.df['Living_Area'] + self.df['Surface_area_plot_of_land']

    def add_amenities_count(self):
        """Adds a feature that counts the total number of amenities."""
        amenities_columns = ['Fully_Equipped_Kitchen', 'Terrace', 'Garden', 'Swimming_Pool', 'Lift']
        if all(col in self.df.columns for col in amenities_columns):
            self.df['Total_Amenities'] = self.df[amenities_columns].sum(axis=1)

    def add_average_room_size(self):
        """Adds a feature for the average room size (Living_Area / Number_of_Rooms)."""
        if 'Living_Area' in self.df.columns and 'Number_of_Rooms' in self.df.columns:
            self.df['Average_Room_Size'] = np.where(self.df['Number_of_Rooms'] > 0, 
                                                     self.df['Living_Area'] / self.df['Number_of_Rooms'], 
                                                     0)

    def add_amenities_ratio(self):
        """Adds a ratio of amenities to the number of rooms."""
        if 'Number_of_Rooms' in self.df.columns and 'Total_Amenities' in self.df.columns:
            self.df['Amenities_Ratio'] = np.where(self.df['Number_of_Rooms'] > 0, 
                                                   self.df['Total_Amenities'] / self.df['Number_of_Rooms'], 
                                                   0)

    def apply_log_transformations(self):
        """
        Applies log transformations to skewed numerical features to reduce skewness.
        Currently applies to 'Living_Area', 'Total_Area', and 'total_income'.
        """
        numerical_features = ['Living_Area', 'Total_Area', 'total_income']
        for feature in numerical_features:
            if feature in self.df.columns:
                self.df[feature + '_log'] = np.log1p(self.df[feature])

    def add_clustered_regions(self, n_clusters=5):
        """
        Adds a clustered region feature based on Locality and Municipality.
        Uses KMeans clustering to assign regions.
        """
        if 'Locality' in self.df.columns and 'Municipality' in self.df.columns:
            combined_regions = self.df[['Locality', 'Municipality']].fillna('missing').astype(str)
            combined_regions_encoded = combined_regions.apply(lambda x: x.astype('category').cat.codes)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.df['Region_Cluster'] = kmeans.fit_predict(combined_regions_encoded)

    def add_interaction_features(self):
        """
        Adds interaction features for improved model performance.
        """
        if 'Distance_to_Nearest_Airport' in self.df.columns and 'Distance_to_Brussels' in self.df.columns:
            self.df['Airport_Brussels_Interaction'] = self.df['Distance_to_Nearest_Airport'] * self.df['Distance_to_Brussels']
        
        if 'Population Density' in self.df.columns and 'Unemployment Rate (%)' in self.df.columns:
            self.df['Density_Unemployment_Ratio'] = self.df['Population Density'] / (self.df['Unemployment Rate (%)'] + 1)
            
    def handle_outliers(self):
        """
        Caps outliers in numerical features to the 1st and 99th percentiles.
        """
        numerical_features = self.df.select_dtypes(include=[np.number]).columns
        for feature in numerical_features:
            lower_bound = self.df[feature].quantile(0.05)
            upper_bound = self.df[feature].quantile(0.95)
            self.df[feature] = np.clip(self.df[feature], lower_bound, upper_bound)
            

    def perform_feature_engineering(self):
        """
        Runs all feature engineering methods to enhance the dataset.
        Returns the transformed DataFrame.
        """
        self.handle_missing_data()          # Handle missing data first
        self.handle_outliers()              # Cap outliers
        self.add_total_area()               # Add total area
        self.add_amenities_count()          # Count total amenities
        self.add_average_room_size()        # Calculate average room size
        self.add_amenities_ratio()          # Compute amenities ratio
        self.apply_log_transformations()    # Log-transform skewed numerical features
        self.add_interaction_features()     # Add interaction features
        self.add_clustered_regions()        # Cluster regions
        return self.df                      # Return the transformed DataFrame

import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder

class ImmowebDataCleaner:
    def __init__(self, data_path):
        """
        Initializes the data cleaner with the provided dataset path.
        """
        self.data_path = data_path
        self.df = None
        self.cat_features = []
        self.num_features = []
        self.target = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def load_data(self):
        """
        Loads the data into a pandas DataFrame with error handling.
        """
        try:
            self.df = pd.read_csv(self.data_path)
            logging.info(f"Data loaded successfully with shape: {self.df.shape}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
    
    def identify_features(self, target_column=None):
        """
        Identifies and classifies the features into categorical and numerical columns.
        Optionally, sets the target column.
        """
        if target_column:
            if target_column not in self.df.columns:
                logging.warning(f"Target column '{target_column}' does not exist in the dataset.")
            else:
                self.target = target_column
        
        # Classify columns
        self.cat_features = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_features = self.df.select_dtypes(include=['number']).columns.tolist()

        if self.target in self.cat_features:
            self.cat_features.remove(self.target)  # Remove target if it's in categorical features
        
        logging.info(f"Categorical Features: {self.cat_features}")
        logging.info(f"Numerical Features: {self.num_features}")
    
    def encode_categorical_data(self, one_hot=False):
        """
        Encodes categorical features for machine learning models.
        If one_hot is True, applies one-hot encoding; otherwise, uses label encoding.
        """
        for col in self.cat_features:
            # Ensure there are no leading/trailing spaces in categorical columns
            self.df[col] = self.df[col].str.strip().fillna('unknown')
        
        if one_hot:
            # Apply OneHot encoding
            self.df = pd.get_dummies(self.df, columns=self.cat_features, drop_first=True)
            logging.info(f"One-Hot Encoding applied to categorical features.")
        else:
            # Apply Label Encoding
            label_encoder = LabelEncoder()
            for col in self.cat_features:
                self.df[col] = label_encoder.fit_transform(self.df[col])
            logging.info(f"Label Encoding applied to categorical features.")
    
    def check_missing_values(self):
        """
        Prints the number of missing values per column.
        """
        missing_values = self.df.isnull().sum()
        logging.info("Missing values per column:")
        logging.info(missing_values)
    
    def prepare_for_catboost(self):
        """
        Prepares the data for CatBoost model by handling categorical features.
        """
        # CatBoost can handle categorical features directly by passing their indices or names
        logging.info("Preparing data for CatBoost...")
        catboost_features = self.cat_features  # List of categorical columns

        # Data is ready for CatBoost.
        logging.info(f"Data is ready for CatBoost. Categorical features: {catboost_features}")
    
    def save_cleaned_data(self, output_path='cleaned_data.csv'):
        """
        Saves the cleaned data to a CSV file.
        """
        self.df.to_csv(output_path, index=False)
        logging.info(f"Cleaned data saved to {output_path}")

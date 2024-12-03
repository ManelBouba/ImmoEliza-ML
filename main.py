from immoweb_data_cleaner_module import ImmowebDataCleaner
from feature_engineering import FeatureEngineering
import pandas as pd

def main():
    # Load your dataset
    df = pd.read_csv('immoweb_with_all_columns.csv')

    # Create an instance of the FeatureEngineering class
    fe = FeatureEngineering(df)

    # Perform feature engineering
    df_transformed = fe.perform_feature_engineering()

    # Save the transformed DataFrame to a new CSV file
    df_transformed.to_csv('immoweb_transformed.csv', index=False)

    data_path = 'immoweb_transformed.csv'
    
    # Initialize the ImmowebDataCleaner and load the data
    data_cleaner = ImmowebDataCleaner(data_path)
    data_cleaner.load_data()
    
    # Perform data cleaning steps
    data_cleaner.identify_features(target_column='Price')
    data_cleaner.check_missing_values()
    data_cleaner.encode_categorical_data()
    
    # Save the cleaned and feature-engineered dataset
    data_cleaner.save_cleaned_data(output_path='immoweb_data_cleaned.csv')

if __name__ == "__main__":
    main()

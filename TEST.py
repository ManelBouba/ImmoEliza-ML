import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import random

# Load the pre-trained model (make sure it's trained and available in the current directory)
model = CatBoostRegressor()
model.load_model("catboost_model_with_tuning.cbm")

# Function to generate random data for the features
def generate_random_data(num_samples=5):
    # Defining categorical features and their possible values
    localities = ['Uccle', 'Koekelare', 'Gembloux', 'Gent', 'Brussels', 'Antwerp', 'Ghent']
    property_types = ['APARTMENT', 'HOUSE', 'VILLA']
    property_subtypes = ['PENTHOUSE', 'APARTMENT', 'HOUSE', 'BUNGALOW']
    states_of_building = ['JUST_RENOVATED', 'GOOD', 'OLD']
    municipalities = ['Uccle', 'Koekelare', 'Gembloux', 'Gent', 'Brussels']
    provinces = ['Brussels Capital', 'Namur', 'East Flanders', 'Other']
    
    # Generate random data
    data = {

        'Locality': [random.choice(localities) for _ in range(num_samples)],
        'Type_of_Property': [random.choice(property_types) for _ in range(num_samples)],
        'Subtype_of_Property': [random.choice(property_subtypes) for _ in range(num_samples)],
        'State_of_the_Building': [random.choice(states_of_building) for _ in range(num_samples)],
        'Number_of_Rooms': [random.randint(1, 5) for _ in range(num_samples)],
        'Living_Area': [random.uniform(50.0, 200.0) for _ in range(num_samples)],
        'Fully_Equipped_Kitchen': [random.choice([0, 1]) for _ in range(num_samples)],  # Categorical (0 or 1)
        'Terrace': [random.choice([0.0, 1.0]) for _ in range(num_samples)],
        'Garden': [random.choice([0.0, 1.0]) for _ in range(num_samples)],
        'Surface_area_plot_of_land': [random.uniform(1.0, 500.0) for _ in range(num_samples)],
        'Number_of_Facades': [random.choice([0.0, 1.0, 2.0]) for _ in range(num_samples)],
        'Swimming_Pool': [random.choice([0.0, 1.0]) for _ in range(num_samples)],
        'Lift': [random.choice([0.0, 1.0]) for _ in range(num_samples)],
        'Municipality': [random.choice(municipalities) for _ in range(num_samples)],
        'Province': [random.choice(provinces) for _ in range(num_samples)],
        'Distance_to_Brussels': [random.uniform(5.0, 100.0) for _ in range(num_samples)],
        'Distance_to_Nearest_Airport': [random.uniform(5.0, 50.0) for _ in range(num_samples)],
        'total_income': [random.uniform(1000000000.0, 10000000000.0) for _ in range(num_samples)],
        'Employment Rate (%)': [random.uniform(50.0, 80.0) for _ in range(num_samples)],
        'Unemployment Rate (%)': [random.uniform(5.0, 20.0) for _ in range(num_samples)],
        'Population Density': [random.uniform(100.0, 5000.0) for _ in range(num_samples)],
        'Total_Area': [random.uniform(50.0, 400.0) for _ in range(num_samples)],
        'Total_Amenities': [random.randint(1, 10) for _ in range(num_samples)],
        'Average_Room_Size': [random.uniform(10.0, 100.0) for _ in range(num_samples)],
        'Amenities_Ratio': [random.uniform(0.1, 1.5) for _ in range(num_samples)],
        'Living_Area_log': [np.log(x) if x > 0 else 0 for x in [random.uniform(50.0, 200.0) for _ in range(num_samples)]],
        'Total_Area_log': [np.log(x) if x > 0 else 0 for x in [random.uniform(50.0, 400.0) for _ in range(num_samples)]],
        'total_income_log': [np.log(x) if x > 0 else 0 for x in [random.uniform(1000000000.0, 10000000000.0) for _ in range(num_samples)]],
        'Airport_Brussels_Interaction': [random.uniform(100.0, 3000.0) for _ in range(num_samples)],
        'Density_Unemployment_Ratio': [random.uniform(10.0, 150.0) for _ in range(num_samples)],
        'Region_Cluster': [random.randint(0, 4) for _ in range(num_samples)]
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)

    return df

# Generate 5 samples of random data
new_data = generate_random_data(num_samples=5)

# Print the generated data to check
print("Generated Data:")
print(new_data)

# Step 2: Make Predictions
# List of categorical features (same as in your model)
cat_features = [
        'Locality', 'Type_of_Property', 'Subtype_of_Property', 
        'State_of_the_Building', 'Fully_Equipped_Kitchen', 
        'Terrace', 'Garden', 'Swimming_Pool', 'Lift', 
        'Municipality', 'Province'
    ]  # Add Fully_Equipped_Kitchen to categorical features

# Ensure categorical features are treated as strings
for feature in cat_features:
    new_data[feature] = new_data[feature].astype(str)

# Handle any missing values if necessary (this dataset should not have any)
new_data = new_data.fillna('Unknown')

# Create the Pool for prediction
new_data_pool = Pool(new_data, cat_features=cat_features)

# Make predictions using the trained model
predictions = model.predict(new_data_pool)

# Exponentiate the predictions to get the original price scale
predicted_prices = np.exp(predictions)

# Output the predicted prices in the original scale
print("Predicted Prices (Original Scale):")
print(predicted_prices)
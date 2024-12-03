import pandas as pd

# Step 1: Read the CSV file (real estate data with distances and airports)
immoweb_data = pd.read_csv('Features/immoweb_with_distances_and_airports.csv')

# Economic Data: Employment and Unemployment Rates for different regions
employment_rate_data = {
    'Belgium': 72.2,
    'Brussels-Capital Region': 63.7,
    'Flemish Region': 76.2,
    'Walloon Region': 68.1
}

unemployment_rate_data = {
    'Belgium': 5.4,
    'Brussels-Capital Region': 12.3,
    'Flemish Region': 3.4,
    'Walloon Region': 6.8
}

# Step 2: Function to get economic data based on region
def get_economic_data(region):
    # Get employment and unemployment rates for a region
    employment_rate = employment_rate_data.get(region, None)
    unemployment_rate = unemployment_rate_data.get(region, None)
    
    return employment_rate, unemployment_rate

# Step 3: Create a function to determine the region based on the postal code
def get_region(postal_code):
    # Ensure the postal code is valid (it should be a string and exactly 4 digits)
    if not isinstance(postal_code, str) or len(postal_code) != 4 or not postal_code.isdigit():
        return 'Invalid postal code'
    
    # Manually assign regions based on postal code
    if postal_code.startswith(('10', '11', '12')):  # Brussels-Capital Region
        return 'Brussels-Capital Region'
    elif postal_code.startswith(('20', '21', '22', '23', '24', '25', '26', '27', '28', '29')):  # Flemish Region
        return 'Flemish Region'
    elif postal_code.startswith(('60', '61', '62', '63', '64', '65', '66', '67', '68', '69')):  # Walloon Region
        return 'Walloon Region'
    else:
        return 'Belgium'  # Default to Belgium for unknown postal codes

# Step 4: Add the economic data to the real estate dataset
def add_economic_features(row):
    # Determine the region based on the postal code (or locality)
    region = get_region(str(row['Locality']))  # Assuming 'locality' is the postal code or region name
    
    # Get the economic data (employment and unemployment rates)
    employment_rate, unemployment_rate = get_economic_data(region)
    
    # Add the economic data as new columns in the real estate data
    row['Employment Rate (%)'] = employment_rate
    row['Unemployment Rate (%)'] = unemployment_rate
    
    return row

# Load the municipality income data
municipality_income_data = pd.read_csv('Features/municipality_total_income_2022.csv')  # This file contains 'Municipality' and 'total_income'
# Rename the column in municipality_income_data to match immoweb_data
municipality_income_data.rename(columns={'TX_MUNTY_DESCR_FR': 'Municipality'}, inplace=True)
# Merge the immoweb data with the municipality income data based on 'Municipality'
immoweb_data = immoweb_data.merge(municipality_income_data, on='Municipality', how='left')

# Optionally, you can save the result to a new file if you want
immoweb_data.to_csv('Features/merged_immoweb_data_with_income.csv', index=False)


# Step 5: Apply the function to each row in the immoweb data to add economic features
immoweb_data = immoweb_data.apply(add_economic_features, axis=1)

# Step 6: Save the final combined dataset to a new CSV file
immoweb_data.to_csv('Features/immoweb_with_all_columns.csv', index=False)

print("All columns have been added and saved to 'immoweb_with_all_columns.csv'.")

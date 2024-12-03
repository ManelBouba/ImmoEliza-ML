import pandas as pd

# Load the dataset
file_path = 'Features/immoweb_with_all_columns.csv'  # Update the path if needed
data = pd.read_csv(file_path)

# Check if 'total_income' column exists in the dataset
data['total_income'] =data.groupby('Province')['total_income'].transform(lambda x: x.fillna(x.mean()))

# Define column categories
numerical_or_binary_columns = [
    'Locality', 'Fully_Equipped_Kitchen', 'Terrace', 'Garden', 
    'Swimming_Pool', 'Lift'
]
categorical_columns = [
    'Type_of_Property', 'Subtype_of_Property', 'State_of_the_Building',
    'Municipality', 'Province'
]

# Convert numerical or binary columns to numeric where applicable
for col in numerical_or_binary_columns:
    if data[col].dtype == 'object':
        try:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        except ValueError:
            print(f"Column `{col}` contains non-numeric values, leaving it as text.")

# Check for missing values
missing_values_summary = data.isnull().sum()
missing_values_percentage = (missing_values_summary / len(data)) * 100
missing_summary = pd.DataFrame({
    'Missing Values': missing_values_summary,
    'Percentage': missing_values_percentage
}).sort_values(by='Percentage', ascending=False)

# Statistical summaries
numerical_summary = data.describe(include=[float, int])
categorical_summary = data[categorical_columns].describe(include=[object])

# Data type analysis
data_types = data.dtypes

# Display results
print("=== Missing Values Summary ===")
print(missing_summary)
print("\n=== Numerical/Binary Columns Summary ===")
print(numerical_summary)
print("\n=== Categorical Columns Summary ===")
print(categorical_summary)
print("\n=== Data Types ===")
print(data_types)

# Save the cleaned dataset if necessary
data.to_csv('immoweb_with_all_columns.csv', index=False)

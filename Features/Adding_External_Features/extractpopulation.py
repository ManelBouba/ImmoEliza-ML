# import pandas as pd

# # Load the Excel file, skipping the first row to use the second row as the header
# excel_file = 'Pop_density_en.xlsx'  # Replace with your Excel file path
# df = pd.read_excel(excel_file, sheet_name=0, skiprows=1)  # Skip the first row

# # Print the column names for debugging
# print("Adjusted Column names:", df.columns.tolist())

# # Save the DataFrame as a CSV file with a semicolon delimiter
# df.to_csv('totalpop.csv', sep=';', index=False)  # Save the file correctly

# # Inspect the DataFrame to ensure the headers and rows are correct
# print("First few rows of the DataFrame:")
# print(df.head())

# # Extract the relevant columns
# if 'Municipality FR' in df.columns and 'Population / km²' in df.columns:
#     # Extract the columns of interest
#     extracted_data = df[['Municipality FR', 'Population / km²']]
    
#     # Rename the columns for clarity
#     extracted_data.columns = ['Municipality', 'Population Density']
    
#     # Save the extracted data to a new CSV file
#     extracted_data.to_csv('extracted_data.csv', index=False)

#     # Display the extracted data
#     print("Extracted Data:")
#     print(extracted_data)
# else:
#     print("Columns 'Municipality FR' and 'Population / km²' not found in the DataFrame.")
import pandas as pd

# Load the immoweb dataset
immoweb_file = 'immoweb_with_all_columns.csv'  # Replace with your file path
immoweb_df = pd.read_csv(immoweb_file)

# Load the extracted data containing population density
extracted_file = 'extracted_data.csv'  # Replace with your file path
extracted_df = pd.read_csv(extracted_file)

# Print both DataFrames for debugging
print("Immoweb DataFrame:")
print(immoweb_df.head())

print("Extracted Population Density DataFrame:")
print(extracted_df.head())

# Ensure consistent naming in both datasets (strip spaces and normalize case)
immoweb_df['Municipality'] = immoweb_df['Municipality'].str.strip().str.lower()
extracted_df['Municipality'] = extracted_df['Municipality'].str.strip().str.lower()

# Merge the extracted population density data into the immoweb dataset
merged_df = immoweb_df.merge(extracted_df, on='Municipality', how='left')

# Save the updated dataset to a new CSV file
merged_df.to_csv('immoweb_with_all_columns.csv', index=False)

# Print the first few rows of the updated dataset
print("Updated Dataset with Population Density:")
print(merged_df.head())

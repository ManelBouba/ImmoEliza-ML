import pandas as pd

# Load the dataset from the Excel file (make sure the file path is correct)
data = pd.read_excel('TF_PSNL_INC_TAX_MUNTY.xlsx')

# Convert the relevant columns to numeric, coercing errors to NaN
income_columns = [
    'MS_TOT_NET_TAXABLE_INC', 'MS_TOT_NET_INC', 'MS_REAL_ESTATE_NET_INC',
    'MS_TOT_NET_MOV_ASS_INC', 'MS_TOT_NET_VARIOUS_INC', 'MS_TOT_NET_PROF_INC'
]

for col in income_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Calculate the total income by summing the relevant income columns
data['total_income'] = data[income_columns].sum(axis=1)

# Filter the data for the year 2022
data_2022 = data[data['CD_YEAR'] == 2022]

# Extract only the relevant columns: municipality and total income
final_data = data_2022[['TX_MUNTY_DESCR_FR', 'total_income']]

# Drop rows with missing municipality descriptions or income
final_data = final_data.dropna(subset=['TX_MUNTY_DESCR_FR', 'total_income'])

# Remove duplicates based on 'municipality' and 'total_income' (if needed)
final_data = final_data.drop_duplicates(subset=['TX_MUNTY_DESCR_FR'])

# Group by 'TX_MUNTY_DESCR_FR' and sum total income
aggregated_data = final_data.groupby('TX_MUNTY_DESCR_FR', as_index=False)['total_income'].sum()

# Save the aggregated data to a CSV file
aggregated_data.to_csv('municipality_total_income_2022.csv', index=False)

print("CSV file with municipality and total income for 2022 has been created.")

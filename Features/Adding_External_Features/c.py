import pandas as pd

# Load the CSV file with the correct delimiter
postal_data = pd.read_csv('postal-codes-belgium.csv', delimiter=';', on_bad_lines='skip')

# Step 2: Clean the column names (remove any hidden characters or extra spaces)
postal_data.columns = postal_data.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True)  # Remove non-ASCII characters
postal_data.columns = postal_data.columns.str.strip()  # Remove extra spaces

# Step 3: Check if 'Geo Point' column exists and is valid
if 'Geo Point' in postal_data.columns:
    print("Geo Point column found.")
    
    # Step 4: Split 'Geo Point' into 'Latitude' and 'Longitude' columns
    # Since the format is "latitude, longitude", split based on the comma
    postal_data[['Latitude', 'Longitude']] = postal_data['Geo Point'].str.split(',', expand=True)

    # Step 5: Clean up spaces in 'Latitude' and 'Longitude'
    postal_data['Latitude'] = postal_data['Latitude'].str.strip()
    postal_data['Longitude'] = postal_data['Longitude'].str.strip()

    # Step 6: Extract Postal Code, Latitude, and Longitude columns
    extracted_data = postal_data[['Postal Code', 'Latitude', 'Longitude']]

    # Step 7: Save the extracted data to a new CSV file
    extracted_data.to_csv('extracted_postal_codes_with_coordinates.csv', index=False)
    print("Extraction complete. New CSV file saved as 'extracted_postal_codes_with_coordinates.csv'.")
else:
    print("Geo Point column not found.")

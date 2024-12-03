import pandas as pd
import math

# Step 1: Read the CSV files
postal_data = pd.read_csv('Features/extracted_postal_codes_with_coordinates.csv')
immoweb_data = pd.read_csv('immoweb_data_cleaned.csv')

# Step 2: Define the Haversine function to calculate the distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Distance in kilometers

# Step 3: Coordinates of Brussels (latitude and longitude)
brussels_lat = 50.8503
brussels_lon = 4.3517

# Step 4: Coordinates of Major Airports in Belgium
airports = {
    'Brussels Airport (BRU)': (50.9014, 4.4844),
    'Charleroi Airport (CRL)': (50.4506, 4.4515),
    'Antwerp Airport (ANR)': (51.1894, 4.4606),
    'Liège Airport (LGG)': (50.6370, 5.4436),
    'Ostend-Bruges Airport (OST)': (51.2130, 2.8641)
}

# Step 5: Remove duplicate postal codes, keeping only the first occurrence
postal_data = postal_data.drop_duplicates(subset=['Postal Code'])

# Step 6: Create a dictionary to map postal codes to their corresponding coordinates
postal_code_map = postal_data.set_index('Postal Code')[['Latitude', 'Longitude']].to_dict(orient='index')

# Step 7: Function to get the coordinates of a postal code
def get_coordinates(postal_code):
    if postal_code in postal_code_map:
        return postal_code_map[postal_code]['Latitude'], postal_code_map[postal_code]['Longitude']
    return None, None  # Return None if the postal code is not found

# Step 8: Function to calculate the distance to Brussels
def calculate_distance_to_brussels(row):
    postal_code = row['Locality']  # Get postal code (locality) from immoweb_data
    lat, lon = get_coordinates(postal_code)
    if lat is not None and lon is not None:
        return haversine(lat, lon, brussels_lat, brussels_lon)
    return None  # If coordinates are missing, return None

# Step 9: Function to calculate the nearest airport
def calculate_nearest_airport(row):
    postal_code = row['Locality']  # Get postal code (locality) from immoweb_data
    lat, lon = get_coordinates(postal_code)
    if lat is not None and lon is not None:
        # Calculate the distance to each airport and find the minimum distance
        min_distance = float('inf')
        for airport, (airport_lat, airport_lon) in airports.items():
            distance = haversine(lat, lon, airport_lat, airport_lon)
            if distance < min_distance:
                min_distance = distance
        return min_distance  # Return the distance to the nearest airport
    return None  # If coordinates are missing, return None

# Step 10: Calculate the distance to Brussels for each property
immoweb_data['Distance_to_Brussels'] = immoweb_data.apply(calculate_distance_to_brussels, axis=1)

# Step 11: Calculate the distance to the nearest airport for each property
immoweb_data['Distance_to_Nearest_Airport'] = immoweb_data.apply(calculate_nearest_airport, axis=1)

# Step 12: Save the updated immoweb data to a new CSV file
immoweb_data.to_csv('Features/immoweb_with_distances_and_airports.csv', index=False)

print("Distance to Brussels and nearest airport added and saved to 'immoweb_with_distances_and_airports.csv'.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the transformed dataset
df_transformed = pd.read_csv('immoweb_transformed.csv')

# 1. Examine the structure of the dataset
print("Basic Information about the dataset:")
print(df_transformed.info())

# Check for any missing values
print("\nMissing values in the dataset:")
print(df_transformed.isnull().sum())

# 2. Basic Statistical Summary for Numerical Columns
print("\nStatistical Summary of Numerical Features:")
print(df_transformed.describe())

# 3. Check for Missing Data After Transformation
missing_values = df_transformed.isnull().sum()
print(f"\nMissing values after transformation:\n{missing_values[missing_values > 0]}")

# 4. Visualizations
sns.set(style="whitegrid")

# Plot 1: Distribution of Price
plt.figure(figsize=(8, 6))
sns.histplot(df_transformed['Price'], kde=True, bins=30, color='blue')
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Plot 2: Price vs. Living Area
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Living_Area', y='Price', data=df_transformed, color='red')
plt.title('Price vs. Living Area')
plt.xlabel('Living Area (m²)')
plt.ylabel('Price')
plt.show()

# Plot 3: Correlation Heatmap (only numerical columns)
plt.figure(figsize=(10, 8))

# Select only numerical columns for correlation
numerical_columns = df_transformed.select_dtypes(include=[float, int]).columns
corr = df_transformed[numerical_columns].corr()

sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Plot 4: Average Price per Locality (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Locality', y='Price', data=df_transformed)
plt.title('Price Distribution by Locality')
plt.xlabel('Locality')
plt.ylabel('Price')
plt.xticks(rotation=90)
plt.show()

# Plot 5: Price per m² vs. Living Area
plt.figure(figsize=(8, 6))
sns.scatterplot(x='price_per_m2', y='Living_Area', data=df_transformed, color='green')
plt.title('Price per m² vs. Living Area')
plt.xlabel('Price per m²')
plt.ylabel('Living Area (m²)')
plt.show()

# Plot 6: Amenities Count vs. Price
plt.figure(figsize=(8, 6))
sns.boxplot(x='amenities_count', y='Price', data=df_transformed, palette='viridis')
plt.title('Price Distribution by Amenities Count')
plt.xlabel('Amenities Count')
plt.ylabel('Price')
plt.show()

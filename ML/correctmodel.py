import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
import numpy as np

# Step 1: Load Data
data_path = "immoweb_data_cleaned.csv"
data = pd.read_csv(data_path)
print(f"Data loaded successfully with shape: {data.shape}")

# Step 2: Define Categorical and Numerical Features
categorical_features = ['Locality', 'Type_of_Property', 'Subtype_of_Property', 
                       'State_of_the_Building', 'Fully_Equipped_Kitchen', 
                       'Terrace', 'Garden', 'Swimming_Pool', 'Lift', 
                       'Municipality', 'Province']
target = 'Price'
X = data.drop(columns=[target])
y = data[target]

# Step 3: Preprocessing Categorical Features
# Ensure categorical columns are converted to string type and clean up any malformed data

# Convert all categorical features to strings
for feature in categorical_features:
    X[feature] = X[feature].astype(str)

# Check for concatenated or problematic entries
# (e.g., entries that are too long or contain unwanted characters)
print(X[categorical_features].head())  # Inspect the first few rows

# If necessary, clean the strings (example: remove unwanted characters)
X[categorical_features] = X[categorical_features].replace({'\n': ' ', '\r': ' ', ',': ''}, regex=True)

# Handle any concatenated or malformed categories by splitting them (if needed)
# Example: If some values are concatenated without spaces, you can split them (based on your dataset structure)
# Adjust this logic based on your data
X['Locality'] = X['Locality'].str.split(' ').str[0]  # Modify based on your data structure if needed

# Ensure no NaN values in categorical columns
X[categorical_features] = X[categorical_features].fillna('missing')

# Step 4: Split Data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Prepare Data Pools for CatBoost
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
valid_pool = Pool(X_valid, y_valid, cat_features=categorical_features)

# Step 6: Train CatBoost Model
model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.1,
    loss_function='RMSE',
    cat_features=categorical_features,
    verbose=100,
    early_stopping_rounds=50
)

print("Training the model...")
model.fit(train_pool, eval_set=valid_pool)

# Step 7: Evaluate Model
print("\nEvaluating the model...")
train_rmse = model.get_best_score()["learn"]["RMSE"]
valid_rmse = model.get_best_score()["validation"]["RMSE"]
print(f"Training RMSE: {train_rmse}")
print(f"Validation RMSE: {valid_rmse}")

# Step 8: Feature Importance
print("\nFeature Importance:")
feature_importances = model.get_feature_importance(train_pool)
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
print(importance_df)

# Optional: Save Feature Importance to CSV
importance_df.to_csv("feature_importance.csv", index=False)

# Step 9: Predictions
print("\nMaking predictions on the validation set...")
predictions = model.predict(X_valid)

# Step 10: Save the Model and Predictions
model.save_model("catboost_model.cbm")
np.savetxt("predictions.csv", predictions, delimiter=",", header="Predicted Prices", comments="")

print("Workflow completed successfully!")

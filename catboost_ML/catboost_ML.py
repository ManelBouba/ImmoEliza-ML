import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from catboost import CatBoostRegressor, Pool
import numpy as np
from scipy.stats import uniform, randint

# Step 1: Load Data
data_path = "Data/immoweb_transformed.csv"  # Ensure your transformed data file is in the correct path
data = pd.read_csv(data_path)
print(f"Data loaded successfully with shape: {data.shape}")

# Step 2: Define Categorical Features
categorical_features = [
    'Locality', 'Type_of_Property', 'Subtype_of_Property', 
    'State_of_the_Building', 'Fully_Equipped_Kitchen', 
    'Terrace', 'Garden', 'Swimming_Pool', 'Lift', 
    'Municipality', 'Province'
]

# Step 3: Handle Missing Values and Convert Categorical Features to String
for feature in categorical_features:
    # Replace NaN values with a string placeholder (e.g., "missing") before converting to string
    data[feature] = data[feature].fillna("missing").astype(str)

# Step 4: Define Target Variable and Features
target = 'Price_log'  # Using the log-transformed price as the target
X = data.drop(columns=['Price', target])  # Drop the original price column
y = data[target]

# Step 5: Split Data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Prepare Data Pools for CatBoost
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
valid_pool = Pool(X_valid, y_valid, cat_features=categorical_features)

# Step 7: Define Hyperparameter Search Space
param_dist = {
    'iterations': randint(800, 3000),  # Boosting rounds (iterations)
    'depth': randint(4, 12),           # Depth of the trees
    'learning_rate': uniform(0.005, 0.05),  # Learning rate
    'l2_leaf_reg': uniform(1, 20),     # Regularization strength
    'border_count': randint(10, 255),  # Number of splits for numerical features
    'bagging_temperature': uniform(0.5, 1.0)  # Bagging diversity
}

# Step 8: Initialize CatBoost Model
model = CatBoostRegressor(
    loss_function='RMSE', 
    cat_features=categorical_features,
    verbose=100, 
    random_state=42,
    early_stopping_rounds=50  # To stop early if validation score doesn't improve
)

# Step 9: Hyperparameter Optimization with RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_dist, 
    n_iter=40,  # Number of iterations for better exploration
    scoring='neg_root_mean_squared_error', 
    cv=3,  # Cross-validation folds
    verbose=2, 
    random_state=42, 
    n_jobs=-1
)

# Perform hyperparameter optimization
print("Running RandomizedSearchCV...")
random_search.fit(X_train, y_train)

# Step 10: Extract Best Model and Hyperparameters
print("Best hyperparameters found: ", random_search.best_params_)
best_model = random_search.best_estimator_

# Evaluate the optimized model
train_rmse = np.sqrt(-random_search.score(X_train, y_train))  # Root Mean Squared Error on training set
valid_rmse = np.sqrt(-random_search.score(X_valid, y_valid))  # Root Mean Squared Error on validation set
print(f"Optimized Training RMSE: {train_rmse}")
print(f"Optimized Validation RMSE: {valid_rmse}")

# Step 11: Feature Importance Analysis
print("\nFeature Importance:")
feature_importances = best_model.get_feature_importance(train_pool)
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
print(importance_df)

# Save Feature Importance to CSV
importance_df.to_csv("catboost_ML/optimized_feature_importance.csv", index=False)

# Step 12: Save the Optimized Model and Predictions
best_model.save_model("catboost_ML/optimized_catboost_model.cbm")
predictions = best_model.predict(X_valid)
np.savetxt("catboost_ML/optimized_predictions.csv", predictions, delimiter=",", header="Predicted Prices", comments="")

print("Hyperparameter optimization and predictions completed successfully!")

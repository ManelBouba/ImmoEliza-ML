import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from catboost import CatBoostRegressor, Pool
import numpy as np
from scipy.stats import uniform, randint

# Step 1: Load Data
data_path = "immoweb_transformed.csv"
data = pd.read_csv(data_path)
print(f"Data loaded successfully with shape: {data.shape}")

# Step 2: Convert categorical features to string
categorical_features = ['Locality', 'Type_of_Property', 'Subtype_of_Property', 
                       'State_of_the_Building', 'Fully_Equipped_Kitchen', 
                       'Terrace', 'Garden', 'Swimming_Pool', 'Lift', 
                       'Municipality', 'Province']

# Convert numeric-like categorical columns (e.g., Locality) into strings
for feature in categorical_features:
    data[feature] = data[feature].astype(str)

# Step 3: Define Target Variable and Features
target = 'Price'
X = data.drop(columns=[target])
y = data[target]

# Step 4: Split Data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Prepare Data Pools for CatBoost
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
valid_pool = Pool(X_valid, y_valid, cat_features=categorical_features)

# Step 6: Define Hyperparameter Search Space
param_dist = {
    'iterations': randint(500, 2000),  # Number of boosting iterations
    'depth': randint(4, 10),           # Depth of trees
    'learning_rate': uniform(0.01, 0.2), # Learning rate
    'l2_leaf_reg': uniform(1, 10),     # Regularization strength
    'border_count': randint(5, 255),   # Number of splits for numerical features
    'bagging_temperature': uniform(0, 1), # Controls the amount of sampling
}

# Step 7: Use RandomizedSearchCV to Optimize Hyperparameters
model = CatBoostRegressor(loss_function='RMSE', cat_features=categorical_features, verbose=100)

# Perform the search
random_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_dist, 
    n_iter=20, 
    scoring='neg_root_mean_squared_error', 
    cv=3, 
    verbose=2, 
    random_state=42, 
    n_jobs=-1
)

# Fit the model with hyperparameter optimization
print("Running RandomizedSearchCV...")
random_search.fit(X_train, y_train)

# Step 8: Best Hyperparameters and Model Evaluation
print("Best hyperparameters found: ", random_search.best_params_)
best_model = random_search.best_estimator_

# Evaluate the optimized model
train_rmse = np.sqrt(-random_search.score(X_train, y_train))
valid_rmse = np.sqrt(-random_search.score(X_valid, y_valid))
print(f"Optimized Training RMSE: {train_rmse}")
print(f"Optimized Validation RMSE: {valid_rmse}")

# Step 9: Feature Importance
print("\nFeature Importance:")
feature_importances = best_model.get_feature_importance(train_pool)
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
print(importance_df)

# Optional: Save Feature Importance to CSV
importance_df.to_csv("optimized_feature_importance.csv", index=False)

# Step 10: Save the Optimized Model and Predictions
best_model.save_model("optimized_catboost_model.cbm")
predictions = best_model.predict(X_valid)
np.savetxt("optimized_predictions.csv", predictions, delimiter=",", header="Predicted Prices", comments="")

print("Hyperparameter optimization and predictions completed successfully!")

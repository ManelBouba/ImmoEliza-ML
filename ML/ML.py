import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
import numpy as np

# Step 1: Load Data
data_path = "immoweb_data_cleaned.csv"
data = pd.read_csv(data_path)
print(f"Data loaded successfully with shape: {data.shape}")

# Step 2: Define Categorical and Numerical Features
categorical_features = ['Type_of_Property', 'Subtype_of_Property', 'State_of_the_Building', 'Municipality', 'Province']
target = 'Price'
X = data.drop(columns=[target])
y = data[target]

# Step 3: Split Data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Prepare Data Pools for CatBoost
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
valid_pool = Pool(X_valid, y_valid, cat_features=categorical_features)

# Step 5: Train CatBoost Model
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

# Step 6: Evaluate Model
print("\nEvaluating the model...")
train_rmse = model.get_best_score()["learn"]["RMSE"]
valid_rmse = model.get_best_score()["validation"]["RMSE"]
print(f"Training RMSE: {train_rmse}")
print(f"Validation RMSE: {valid_rmse}")

# Step 7: Feature Importance
print("\nFeature Importance:")
feature_importances = model.get_feature_importance(train_pool)
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
print(importance_df)

# Optional: Save Feature Importance to CSV
importance_df.to_csv("ML/feature_importance.csv", index=False)

# Step 8: Predictions
print("\nMaking predictions on the validation set...")
predictions = model.predict(X_valid)

# Step 9: Save the Model and Predictions
model.save_model("ML/catboost_model.cbm")
np.savetxt("ML/predictions.csv", predictions, delimiter=",", header="Predicted Prices", comments="")

print("Workflow completed successfully!")

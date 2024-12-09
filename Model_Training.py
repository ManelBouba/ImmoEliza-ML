import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

class ModelTraining:
    def __init__(self, X, y, categorical_features):
        self.X = X
        self.y = y
        self.categorical_features = categorical_features

    def split_data(self):
        """Split the dataset into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_model(self):
        """Train the model using CatBoostRegressor."""
        best_params = {
            'learning_rate': 0.01,
            'iterations': 5000,
            'depth': 6,
            'border_count': 32,
            'bagging_temperature': 0.1,
            'l2_leaf_reg': 15
        }

        self.model = CatBoostRegressor(
            learning_rate=best_params['learning_rate'],
            l2_leaf_reg=best_params['l2_leaf_reg'],
            iterations=best_params['iterations'],
            depth=best_params['depth'],
            border_count=best_params['border_count'],
            bagging_temperature=best_params['bagging_temperature'],
            loss_function='RMSE', 
            cat_features=self.categorical_features,
            random_state=42,
            verbose=200,
            early_stopping_rounds=50
        )

        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Evaluate the model's performance on the training and testing sets."""
        # Train predictions
        y_train_pred = self.model.predict(self.X_train)
        y_train_pred = np.expm1(y_train_pred)  # Reverse log1p on predictions

        # Test predictions
        y_test_pred = self.model.predict(self.X_test)
        y_test_pred = np.expm1(y_test_pred)  # Reverse log1p on predictions

        # Training metrics
        train_rmse = np.sqrt(mean_squared_error(np.expm1(self.y_train), y_train_pred))
        train_mae = mean_absolute_error(np.expm1(self.y_train), y_train_pred)
        train_r2 = r2_score(np.expm1(self.y_train), y_train_pred)
        train_mape = np.mean(np.abs((np.expm1(self.y_train) - y_train_pred) / np.expm1(self.y_train))) * 100
        train_smape = self.smape(np.expm1(self.y_train), y_train_pred)

        # Test metrics
        test_rmse = np.sqrt(mean_squared_error(np.expm1(self.y_test), y_test_pred))
        test_mae = mean_absolute_error(np.expm1(self.y_test), y_test_pred)
        test_r2 = r2_score(np.expm1(self.y_test), y_test_pred)
        test_mape = np.mean(np.abs((np.expm1(self.y_test) - y_test_pred) / np.expm1(self.y_test))) * 100
        test_smape = self.smape(np.expm1(self.y_test), y_test_pred)

        # Print training and test performance metrics
        print(f"Training RMSE: {train_rmse}")
        print(f"Test RMSE: {test_rmse}")
        print(f"Training MAE: {train_mae}")
        print(f"Test MAE: {test_mae}")
        print(f"Training R²: {train_r2}")
        print(f"Test R²: {test_r2}")
        print(f"Training MAPE: {train_mape}%")
        print(f"Test MAPE: {test_mape}%")
        print(f"Training sMAPE: {train_smape}%")
        print(f"Test sMAPE: {test_smape}%")

        return {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_mape": train_mape,
            "test_mape": test_mape,
            "train_smape": train_smape,
            "test_smape": test_smape
        }

    def smape(self, y_true, y_pred):
        """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)."""
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    def feature_importance(self):
        """Extract and print the feature importance."""
        importances = self.model.get_feature_importance()
        importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        print("\nFeature Importance:")
        print(importance_df)

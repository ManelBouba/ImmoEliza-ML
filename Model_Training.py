# model_training.py
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))

        print(f"Training RMSE: {train_rmse}")
        print(f"Test RMSE: {test_rmse}")

        return train_rmse, test_rmse

    def model_performance_metrics(self):
        """Print additional evaluation metrics like MAE and R²."""
        y_pred = self.model.predict(self.X_test)
        y_pred = np.expm1(y_pred)  # Reverse log1p on predictions

        test_mae = mean_absolute_error(np.expm1(self.y_test), y_pred)
        test_r2 = r2_score(np.expm1(self.y_test), y_pred)

        print(f"Test MAE: {test_mae}")
        print(f"Test R²: {test_r2}")

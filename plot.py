# plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
import pandas as pd
class Plotting:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def plot_residuals(self):
        """Plot residuals to evaluate the model's predictions."""
        y_pred = self.model.predict(self.X_test)
        y_pred = np.expm1(y_pred)  # Inverse log transformation on predictions

        residuals = np.expm1(self.y_test) - y_pred

        print(f"Residuals Statistics:")
        print(f"Mean of residuals: {np.mean(residuals)}")
        print(f"Standard deviation of residuals: {np.std(residuals)}")
        print(f"Max residual: {np.max(residuals)}")
        print(f"Min residual: {np.min(residuals)}")

        # Plot residuals vs predicted values
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted Values")
        plt.show()

        # Plot the distribution of residuals
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=50)
        plt.title("Residual Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.show()

    def plot_feature_importance(self, feature_names):
        """Plot the feature importance to understand the model's decision-making process."""
        importances = self.model.get_feature_importance()
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Print feature importance
        print("\nFeature Importance:")
        print(importance_df)
        # Plot the feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title("Feature Importances")
        plt.show()

    def plot_shap_values(self, explainer, shap_values):
        """Generate SHAP visualizations."""
        # Plot summary plot
        print("Generating SHAP summary plot...")
        shap.summary_plot(shap_values, self.X_test)

        # Plot feature importance bar plot
        print("Generating SHAP bar plot...")
        shap.summary_plot(shap_values, self.X_test, plot_type="bar")

        # Visualize a single prediction
        print("Visualizing a single prediction with SHAP...")
        shap.force_plot(explainer.expected_value, shap_values[0, :], self.X_test.iloc[0, :])

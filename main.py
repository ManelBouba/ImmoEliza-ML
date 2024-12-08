import pandas as pd
import numpy as np
from Feature_Engineering import FeatureEngineering
from Data_cleaning import DataCleaning
from Model_Training import ModelTraining
from plot import Plotting
import shap

def main():
    # Step 1: Load the dataset
    df = pd.read_csv("Data/immoweb_with_all_columns.csv")

    # Step 2: Apply Feature Engineering
    feature_engineering = FeatureEngineering(df)
    df_transformed = feature_engineering.perform_feature_engineering()

    # Save the transformed dataset
    df_transformed.to_csv("Data/immoweb_features2.csv", index=False)

    # Step 3: Data Cleaning
    data_cleaner = DataCleaning('Data/immoweb_features2.csv')
    df_cleaned = data_cleaner.clean_data()
    X, y = data_cleaner.get_features_and_target()

    # Step 4: Model Training
    categorical_features = [
        'Locality', 'Type_of_Property', 'Subtype_of_Property', 
        'State_of_the_Building', 'Fully_Equipped_Kitchen', 
        'Terrace', 'Garden', 'Swimming_Pool', 'Lift', 
        'Municipality', 'Province'
    ]
    model_trainer = ModelTraining(X, y, categorical_features)
    model_trainer.split_data()
    model_trainer.train_model()

    # Step 5: Model Evaluation
    train_rmse, test_rmse = model_trainer.evaluate_model()
    model_trainer.model_performance_metrics()

    # Step 6: Plotting
    plotter = Plotting(model_trainer.model, model_trainer.X_test, model_trainer.y_test)
    plotter.plot_residuals()
    plotter.plot_feature_importance(model_trainer.X_train.columns)

    # Step 7: SHAP Analysis
    explainer = shap.TreeExplainer(model_trainer.model)
    shap_values = explainer.shap_values(model_trainer.X_test)
    plotter.plot_shap_values(explainer, shap_values)

    # Step 8: Save Model and Predictions
    model_path = 'catboost_model_with_tuning.cbm'
    predictions_path = 'predictions.csv'
    model_trainer.model.save_model(model_path)
    np.savetxt(predictions_path, np.expm1(model_trainer.model.predict(model_trainer.X_test)), delimiter=",", header="Predicted Prices", comments="")

    print(f"Model saved to {model_path}")
    print(f"Predictions saved to {predictions_path}")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()

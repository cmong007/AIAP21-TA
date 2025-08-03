import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def load_and_prepare_data(file_path: str) -> (pd.DataFrame, pd.Series, LabelEncoder):
    """
    Loads data from a CSV, performs feature engineering and selection based on EDA,
    and encodes the target variable.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        exit()

    # Feature Engineering and Selection
    df['MetalOxideSensor_Unit1&4_avg'] = df[['MetalOxideSensor_Unit1', 'MetalOxideSensor_Unit4']].mean(axis=1)
    col_to_drop = ["HVAC Operation Mode", "Ambient Light Level", "Session ID", "MetalOxideSensor_Unit1", "MetalOxideSensor_Unit4"]
    df = df.drop(columns=col_to_drop, axis=1)

    X = df.drop("Activity Level", axis=1)
    y = df["Activity Level"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("Data loaded and prepared successfully.")
    return X, y_encoded, le

def get_preprocessors(X: pd.DataFrame) -> dict:
    """
    Creates and returns a dictionary of preprocessors for different model types.
    """
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor_ohe = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    preprocessor_ordinal = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ])
    
    return {"ohe": preprocessor_ohe, "ordinal": preprocessor_ordinal}

def get_models_and_params() -> (dict, dict):
    """
    Defines and returns the models to be trained and their hyperparameter grids.
    """
    param_grids = {
        "Logistic Regression": {
            'classifier__C': [0.1, 1.0, 10.0]
        },
        "Random Forest": {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20]
        },
        "XGBoost": {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [3, 5]
        }
    }

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss')
    }
    return models, param_grids

def run_evaluation_pipeline(X_train, X_test, y_train, y_test, le, models, param_grids, preprocessors):
    """
    Runs the main training, tuning, and evaluation pipeline for all models.
    """
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_summary = []

    for name, model in models.items():
        print(f"\n{'='*25} {name} {'='*25}")

        preprocessor = preprocessors['ordinal'] if name == "Random Forest" else preprocessors['ohe']

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        grid_search = GridSearchCV(pipeline, param_grids[name], cv=cv_strategy, n_jobs=-1, scoring='accuracy')

        print("Tuning hyperparameters...")
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # --- Predictions ---
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # --- Build Results Dictionary Dynamically ---
        model_results = {"Model": name}
        train_report_dict = classification_report(y_train, y_train_pred, output_dict=True, zero_division=0)
        test_report_dict = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
        
        metrics_to_log = ['precision', 'recall', 'f1-score']
        for metric in metrics_to_log:
            metric_display_name = metric.replace('_', ' ').title()
            model_results[f"Train Macro {metric_display_name}"] = train_report_dict['macro avg'][metric]
            model_results[f"Test Macro {metric_display_name}"] = test_report_dict['macro avg'][metric]

        model_results["Train ROC-AUC"] = roc_auc_score(y_train, best_model.predict_proba(X_train), multi_class='ovr')
        model_results["Test ROC-AUC"] = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')
        
        results_summary.append(model_results)

        # --- Print Individual Model Report ---
        print(f"\nBest Hyperparameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_test_pred, target_names=le.classes_, zero_division=0))

        # --- Confusion Matrix Visualization ---
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix for {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Save the plot for the best model to be included in README
        if name == "Random Forest":
            images_dir = 'images'
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            plt.savefig(os.path.join(images_dir, 'random_forest_confusion_matrix.png'))
            print(f"Confusion matrix for {name} saved to '{images_dir}' directory.")

        plt.show()

    return pd.DataFrame(results_summary)

def main():
    """Main function to orchestrate the script execution."""
    FILE_PATH = "data/gas_monitoring_cleaned.csv"
    
    X, y_encoded, le = load_and_prepare_data(FILE_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    preprocessors = get_preprocessors(X)
    models, param_grids = get_models_and_params()
    
    summary_df = run_evaluation_pipeline(X_train, X_test, y_train, y_test, le, models, param_grids, preprocessors)

    # --- Display Final Consolidated View ---
    print(f"\n{'='*35} Model Comparison Summary {'='*35}")
    summary_df.set_index("Model", inplace=True)
    float_cols = summary_df.select_dtypes(include=['float64']).columns
    summary_df[float_cols] = summary_df[float_cols].map(lambda x: f"{x:.4f}")
    print(summary_df)
    print(f"\n{'='*90}\n")

if __name__ == "__main__":
    main()


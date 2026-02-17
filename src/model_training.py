# This is the main script for training the model. It loads the data, performs feature engineering,
# and trains a machine learning model to predict question quality based on score thresholds.

import time
import pandas as pd
import json
import yaml
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier # For our classification model
from sklearn.linear_model import LogisticRegression # For a simpler baseline model
from sklearn.metrics import ( # For evaluating model performance
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
from src.ingestion import get_vectorized_inputs_and_label # To prepare features and labels from the cleaned data
from src.pipeline_orchestrator import PipelineOrchestrator # To use the feature engineering pipeline for transforming the data before training

# This script orchestrates the entire model training process, from loading the configuration and data to training the model and evaluating its performance. 
# It is designed to be modular and extensible, allowing for easy adjustments to the model architecture, feature engineering steps, and evaluation metrics as needed. 
# The trained model and preprocessing pipeline are saved as artifacts for future inference on new questions.
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_model(config):
    """Factory function to create model instances based on configuration.
    
    Args:
        config: Configuration dictionary containing model settings
    
    Returns:
        Instantiated scikit-learn model
    """
    algorithm = config['model']['algorithm'] # Get the algorithm name from config to determine which model to instantiate
    random_state = config['model']['random_state'] # Get the random state from config for reproducibility of results across runs. 
    
    if algorithm == 'random_forest': # If the specified algorithm is 'random_forest', we create and return an instance of RandomForestClassifier with parameters defined in the config. We also set n_jobs=-1 to utilize all available CPU cores for faster training.
        rf_params = config['model'].get('random_forest', {}) # Get the random forest parameters from config, or use an empty dict if not specified
        return RandomForestClassifier( # Instantiate the Random Forest model with parameters from config
            n_estimators=rf_params.get('n_estimators', 100), # Number of trees in the forest, default is 100 if not specified in config
            max_depth=rf_params.get('max_depth', None), # Maximum depth of the tree, default is None (nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples)
            min_samples_split=rf_params.get('min_samples_split', 2), # Minimum number of samples required to split an internal node, default is 2
            class_weight=rf_params.get('class_weight', None), # Adjusts weights inversely proportional to class frequencies when set to 'balanced'
            oob_score=rf_params.get('oob_score', False), # Use out-of-bag samples to estimate generalization accuracy
            random_state=random_state, # Set the random state for reproducibility
            n_jobs=-1,  # Use all available cores
            verbose=2  # Print progress per tree
        )
    elif algorithm == 'logistic_regression': # If the specified algorithm is 'logistic_regression', we create and return an instance of LogisticRegression with parameters defined in the config. We also set n_jobs=-1 to utilize all available CPU cores for faster training.
        return LogisticRegression( # Instantiate the Logistic Regression model with parameters from config
            random_state=random_state, # Set the random state for reproducibility
            max_iter=1000, # Set a higher max_iter to ensure convergence, especially if we have a large number of features after vectorization
            n_jobs=-1,  # Use all available cores
            verbose=1  # Print convergence progress
        )   
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: 'random_forest', 'logistic_regression'")


def load_and_prepare_data(config):
    """Load train and test data and prepare features and labels.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, median_threshold)
    """
    print("\n" + "="*60) # Print a header for the data loading section to improve readability of the console output.
    print("LOADING DATA") 
    print("="*60)
    # This section is responsible for loading the training and test datasets from the paths specified in the configuration file. It also prepares the features and labels for model training by calling the get_vectorized_inputs_and_label function, which processes the raw data and applies the necessary transformations to create the feature matrix (X) and target vector (y). The median threshold used for labeling is also returned for use in future inference on new questions.
    
    
    # Load training data
    train_path = config['paths']['train_csv'] # Get the training data path from config.yaml to load the training dataset. 
    test_path = config['paths']['test_csv'] # Get the test data path from config.yaml to load the test dataset.
    
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"✅ Train data loaded: {train_df.shape}")
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"✅ Test data loaded: {test_df.shape}")
    
    # Prepare features and labels
    print("\nPreparing features and labels...")
    
    # Use quality-based labels instead of score threshold
    X_train, y_train, median_threshold = get_vectorized_inputs_and_label(train_df, use_quality_label=True)
    print(f"✅ Training set prepared: {X_train.shape}")
    print(f"   Label distribution: {y_train.value_counts().to_dict()}")
    
    # Only print median threshold if it exists (score-based labeling)
    if median_threshold is not None:
        print(f"   Median score threshold: {median_threshold:.2f}")
    else:
        print(f"   Using quality-based labels (ClosedDate, AcceptedAnswerId, Score)")
    
    # Use same quality labeling for test data
    X_test, y_test, _ = get_vectorized_inputs_and_label(test_df, use_quality_label=True)
    print(f"✅ Test set prepared: {X_test.shape}")
    print(f"   Label distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, y_train, X_test, y_test, median_threshold


def fit_feature_pipeline(X_train, X_test, config):
    """Fit feature engineering pipeline on training data and transform both sets.
    
    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        config: Configuration dictionary
    
    Returns:
        Tuple of (X_train_transformed, X_test_transformed, fitted_pipeline)
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    # Initialize pipeline orchestrator
    orchestrator = PipelineOrchestrator(config=config)
    pipeline = orchestrator.get_pipeline()
    
    print(f"Feature configuration:")
    print(f"  - Numeric: {config['features']['numeric']}")
    print(f"  - Categorical: {config['features']['categorical']}")
    print(f"  - Text: {config['features']['text']}")
    print(f"  - Quality features: {config['features']['use_quality_features']}")
    
    # Fit on training data AND transform in a single call.

    print("\nFitting feature pipeline on training data...")
    X_train_transformed = pipeline.fit_transform(X_train)
    print("✅ Pipeline fitted")
    # We can also print the feature names after fitting to verify that the pipeline is processing the features as expected. This is especially useful for debugging and ensuring that all intended features are included in the model training.
    if hasattr(pipeline, 'get_feature_names_out'):
        feature_names = pipeline.get_feature_names_out()
        print(f"Pipeline output feature names: {feature_names}")
    else:
        print("Pipeline does not support get_feature_names_out, skipping feature name display.")

    print(f"✅ Train features shape after transformation: {X_train_transformed.shape}")

    # Transform test data using the already-fitted pipeline.
    # This runs transform() only (no fitting), so the test set is vectorized and processed using the same transformations learned from the training data, 
    # ensuring that the model sees consistent feature representations during both training and evaluation.
    print("Transforming test data...")
    X_test_transformed = pipeline.transform(X_test)
    print(f"✅ Test features shape after transformation: {X_test_transformed.shape}")
    
    return X_train_transformed, X_test_transformed, pipeline


def train_model(X_train, y_train, config):
    """Train the machine learning model.
    
    Args:
        X_train: Transformed training features
        y_train: Training labels
        config: Configuration dictionary
    
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    model = get_model(config)
    print(f"Training {config['model']['algorithm']} model...")
    print(f"Model configuration: {model.get_params()}")
    
    model.fit(X_train, y_train) # We fit the model on the transformed training data (X_train) and the corresponding labels (y_train). 
    #This is where the machine learning algorithm learns the patterns in the data that relate the features to the target variable (question quality in this case). The fitted model can then be used to make predictions on new, unseen data.
    print("✅ Model training completed")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set.
    
    Args:
        model: Trained model
        X_test: Transformed test features
        y_test: Test labels
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test) # We use the trained model to make predictions on the transformed test features (X_test). This generates a predicted label for each instance in the test set, which we can then compare to the true labels (y_test) to evaluate the model's performance.
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) # Accuracy measures the overall correctness of the model's predictions, calculated as the ratio of correctly predicted instances to the total instances in the test set.
    precision = precision_score(y_test, y_pred) # Precision measures the proportion of positive identifications that were actually correct, calculated as the ratio of true positives to the sum of true positives and false positives.
    recall = recall_score(y_test, y_pred) # Recall measures the proportion of actual positives that were correctly identified, calculated as the ratio of true positives to the sum of true positives and false negatives.
    f1 = f1_score(y_test, y_pred) # F1-Score is the harmonic mean of precision and recall, providing a single metric that balances both concerns.
    
    print("\n📊 Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Quality', 'High Quality']))
    
    print("\n🔲 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred) # Confusion matrix shows the counts of true positive, true negative, false positive, and false negative predictions, providing a detailed view of the model's performance.
    print(f"                Predicted")
    print(f"               Low   High")
    print(f"Actual Low    {cm[0][0]:5d} {cm[0][1]:5d}")
    print(f"       High   {cm[1][0]:5d} {cm[1][1]:5d}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def save_artifacts(pipeline, model, median_threshold, config):
    """Save trained model and preprocessing artifacts."""
    output_dir = Path(config['model']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'model.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save pipeline
    pipeline_path = output_dir / 'preprocessing_pipeline.pkl'
    joblib.dump(pipeline, pipeline_path)
    print(f"✅ Pipeline saved to: {pipeline_path}")
    
    # Save metadata
    metadata = {
        'model_type': config['model']['algorithm'],
        'label_type': 'quality_based' if median_threshold is None else 'score_based',
        'median_threshold': median_threshold,
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {metadata_path}")


def run_training_pipeline(config_path="config.yaml"):
    """Main orchestrator for the model training pipeline.
    
    Args:
        config_path: Path to configuration YAML file
    """
    print("\n" + "="*60)
    print("QUESTION QUALITY PREDICTION - MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load configuration
    # Check if we were passed a path (str) or the actual dictionary
    if isinstance(config_path, str):
        config = load_config(config_path)
    else:
        config = config_path # It's already a dict, just use it
    
    # Step 1: Load and prepare data
    t0 = time.time()
    X_train, y_train, X_test, y_test, median_threshold = load_and_prepare_data(config)
    print(f"⏱  Data loading took {time.time() - t0:.1f}s")
    
    # Step 2: Fit feature engineering pipeline
    t0 = time.time()
    X_train_transformed, X_test_transformed, pipeline = fit_feature_pipeline(
        X_train, X_test, config
    )
    print(f"⏱  Feature engineering took {time.time() - t0:.1f}s")
    
    # Step 3: Train model
    t0 = time.time()
    model = train_model(X_train_transformed, y_train, config)
    print(f"⏱  Model training took {time.time() - t0:.1f}s")
    
    # Step 4: Evaluate model
    t0 = time.time()
    metrics = evaluate_model(model, X_test_transformed, y_test)
    print(f"⏱  Evaluation took {time.time() - t0:.1f}s")
    
    # Step 5: Save artifacts
    save_artifacts(pipeline, model, median_threshold, config)
    
    print("\n" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nFinal F1-Score: {metrics['f1_score']:.4f}")
    print(f"Model saved for inference on new questions.")
    

if __name__ == "__main__":
    run_training_pipeline()



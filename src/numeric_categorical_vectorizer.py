#script that normalizes tabular data for machine learning tasks
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.impute import SimpleImputer # To handle missing values in numeric features

# Custom transformer to extract date features
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    # Fit does nothing, just returns self
    def fit(self, X, y=None):
        return self
    # Transform date columns into separate year, month, day features
    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            date_col = pd.to_datetime(X[col])
            X[f'{col}_year'] = date_col.dt.year
            X[f'{col}_month'] = date_col.dt.month
            X[f'{col}_day'] = date_col.dt.day
            X = X.drop(columns=[col]) # Drop the original string column
        return X
    
    # Add method to get feature names after transformation
    def get_feature_names_out(self, input_features=None):
        feature_names = []
        # For every input column, we are creating 3 new ones
        for col in input_features:
            feature_names.extend([f"{col}_year", f"{col}_month", f"{col}_day"])
        return np.array(feature_names)


# Class to create a normalization pipeline 
class NormalizationPipelineFactory:
    def __init__(self):
        pass

    def get_preprocessor(self, feature_config):
        """
        Creates the ColumnTransformer for structured data (Num, Cat, Date).
        
        Args:
            feature_config (dict): A dictionary containing feature lists. 
                                   Example: config['features']
        """
        # Extract lists safely (default to empty list if missing)
        numeric_features = feature_config.get('numeric') or []
        categorical_features = feature_config.get('categorical') or []
        date_features = feature_config.get('date') or []

        print(f"   - Structured Features: {len(numeric_features)} Num, {len(categorical_features)} Cat, {len(date_features)} Date")

        # We will build a list of transformers based on which feature types are present. This allows us to be flexible and only include the transformations we need based on the config.
        transformers = []

        # --- A. Numeric Pipeline ---
        if numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), 
                ('scaler', StandardScaler()) # We use StandardScaler to normalize numeric features to have mean=0 and std=1, which is a common practice for many ML algorithms.
            ])
            transformers.append(('num', numeric_transformer, numeric_features)) # We add this transformer to the list with a name 'num' and specify which columns it should be applied to (numeric_features).

        # --- B. Categorical Pipeline ---
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # Handle missing values in categorical features by filling them with a constant value 'missing'. This ensures that the OneHotEncoder can process the data without errors due to null values.
                ('encoder', OneHotEncoder(
                    handle_unknown='infrequent_if_exist', 
                    max_categories=5,  # Consider moving this to config if you want flexibility
                    sparse_output=False
                ))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))

        # --- C. Date Pipeline ---
        if date_features:
            transformers.append(('date', DateFeatureExtractor(), date_features))

        # --- D. Assemble ---
        # We return the ColumnTransformer directly. 
        # This allows the Orchestrator to wrap it however it wants.
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop' 
        )
        
        return preprocessor
    
    


# initialize and test the pipeline
if __name__ == "__main__":
    # Load the config to get the feature lists
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    feature_config = config['features']
    print(f"Feature Config: {feature_config}")

    # Load the data from the config.yaml file path
    input_path = config['paths']['interim_csv']
    df = pd.read_csv(input_path)
    print(f"✅ Loaded data with {len(df)} rows.")

    # Initialize the NormalizationPipelineFactory and get the preprocessor
    normalization_factory = NormalizationPipelineFactory()
    preprocessor = normalization_factory.get_preprocessor(feature_config)

    # Fit-transform the data using the preprocessor and print the shape of the output
    normalized_data = preprocessor.fit_transform(df)
    print(f"Normalized Data Shape: {normalized_data.shape}")
    #print the first 5 rows of the normalized data to verify it worked
    print("First 5 rows of normalized data:")
    print(normalized_data[:5])
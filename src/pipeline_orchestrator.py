import yaml
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD # For dimensionality reduction if compression is enabled
from sklearn.preprocessing import StandardScaler # For numeric feature normalization

# Local imports
from src.text_vectorizer import TextVectorizer
from src.numeric_categorical_vectorizer import NormalizationPipelineFactory
from src.feature_engineering import QualityFeatureExtractor #include our custom feature extractor in the pipeline to add those features to our dataset before clustering. 

# This class orchestrates the entire pipeline from loading config, building the pipeline, and executing it. It serves as the main entry point for running the data processing and vectorization steps. 
# It combines the text vectorization and numeric/categorical normalization into a single pipeline that can be easily executed. 
# This class can be extended in the future to include more stages or different types of processing as needed.

class PipelineOrchestrator:
    def __init__(self, config_path="config.yaml", config=None):
        # --- 1. Load Config  ---
        if config is not None:
            self.config = config
        else:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

    def load_data(self, path_key='interim_csv'):
        """
        Loads data from the path specified in config by 'path_key'.
        Performs initial filtering and cleaning (Tags).
        """
        # --- 2. Load Data ---
        input_path = self.config['paths'][path_key]
        print(f"📥 Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        
        # Filter for questions only (if needed)
        if 'PostTypeId' in df.columns:
            df = df[df['PostTypeId'] == 1]
        
        print(f"✅ Loaded data with {len(df)} rows.")

        # clean the [], | and <>from Tags to avoid impacting the text vector
        if 'Tags' in df.columns:
            print("🧹 Cleaning brackets from Tags...")
            # Replaces [ and ] with a space, then trims extra whitespace
            df['Tags'] = df['Tags'].str.replace(r'[\[\]]', ' ', regex=True).str.strip()
            # Replaces < or > with a space, then trims extra whitespace
            df['Tags'] = df['Tags'].str.replace(r'[<>]', ' ', regex=True).str.strip()
            # replaces | with a space, then trims extra whitespace
            df['Tags'] = df['Tags'].str.replace(r'[|]', ' ', regex=True).str.strip()
            # print the first 5 rows of the cleaned Tags column to verify it worked
            print(df['Tags'].head())
            
        return df

    def get_pipeline(self):
        """
        Constructs the full preprocessing pipeline but DOES NOT fit it.
        Returns: sklearn.compose.ColumnTransformer wrapped in a Pipeline
        """
        config = self.config

        # --- 3. Build the text Pipeline ---
        # Get feature lists from config and flatten it
        text_features = config['features'].get('text', [])
        text_features = list(set(text_features)) # ensure the uniqueness for text feature
        if any(isinstance(i, list) for i in text_features):
            text_features = [item for sublist in text_features for item in (sublist if isinstance(sublist, list) else [sublist])]
            
        # Get settings from the config section for the text vectorizer
        text_vectorizer_config = config['text_vectorizer']
        model_name = text_vectorizer_config['model']
        
        # Initialize the TextVectorizer with the model specified in the config
        text_vec = TextVectorizer(model=model_name)
        
        # check if compression is enabled in the config 
        if text_vectorizer_config['compression']['enabled']:
            # If compression is enabled, we will add a TruncatedSVD step to the pipeline after the text vectorization to reduce the dimensionality of the output vectors. 
            # This is useful for improving performance and reducing noise in the data while retaining as much information as possible.
            n_components = text_vectorizer_config['compression']['n_components']
            print(f"⚡ Compression enabled: Reducing to {n_components} dimensions using SVD.")
            
            # We can create a separate pipeline for the text vectorization and compression steps, 
            # and then include that in the ColumnTransformer. 
            # This keeps our code organized and allows us to easily toggle compression on or off based on the config.
            text_pipeline = Pipeline(steps=[
                ('vectorizer', text_vec),
                ('svd', TruncatedSVD(n_components=n_components, random_state=42))
            ])
        else:
            print("⚡ Compression disabled: Using full vector dimensions.")
            text_pipeline = Pipeline(steps=[
                ('vectorizer', text_vec)
            ])        

        # 4. build the full pipeline with both text vectorization and numeric/categorical normalization
        # Initialize your NormalizationPipelineFactory
        normalization_factory = NormalizationPipelineFactory()
        
        # Get the config features for numeric, categorical, and date
        feature_config = []
        target_categories = ['numeric', 'categorical', 'date']
        
        # robust logic to go through the list of feature in the config.yaml
        for cat in target_categories:
            # Get the value from config, ensure it is a list
            items = config['features'].get(cat)
            
            if not items: # Skip if None or empty
                continue
                
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, list):
                        feature_config.extend(item) # Handle nested lists
                    else:
                        feature_config.append(item) # Handle strings in list
            else:
                feature_config.append(items) # Handle single string edge case
        feature_config = list(set(feature_config)) # ensure uniqueness of the list

        # Define the ColumnTransformer to apply different transformations to different columns of your dataset simultaneously. 
        # We specify that the text vectorizer should be applied to the text features defined in the config. 
        # We can add additional transformers for numeric and date features as needed, following a similar pattern.

        # We initialise a transformer list with the text pipeline, and the normalization pipeline.
        transformer_list = [
            ('text_processing', text_pipeline, text_features),
            ('structured_processing', normalization_factory.get_preprocessor(config['features']), feature_config)
        ]

        # If the config has use_quality_features set to true, we will include our custom feature extractor in the pipeline to add those features to our dataset before clustering. 
        if config['features'].get('use_quality_features', False):
            print("✨ Adding quality features to the pipeline...")
            quality_feature_extractor = QualityFeatureExtractor()
            quality_pipeline = Pipeline(steps=[
                ('quality_features', quality_feature_extractor),
                ('scaler', StandardScaler()) # We add a scaler to normalize the quality features since they may be on a different scale than the other features.
            ])
        
            # append qualiy_pipeline to the transformer list if use_quality_features is true
            transformer_list.append(('quality_processing', quality_pipeline, ['Title', 'Body']))
        
        # We then create the ColumnTransformer with the list of transformers we have defined. This allows us to apply all the transformations in one step when we fit_transform the pipeline.
        preprocessor = ColumnTransformer(
            transformers=transformer_list,
            remainder='drop' # We drop any columns that are not specified in the transformers list. This ensures that only the features we want to include in the clustering process are retained in the output matrix.
        )
        
        # We wrap the ColumnTransformer in a Pipeline. This allows us to easily add more steps in the future if needed (e.g., feature selection, dimensionality reduction, etc.) and keeps our code organized.
        # NOTE: We return the pipeline object itself, NOT the result of fit_transform.
        return Pipeline(steps=[('preprocessor', preprocessor)])

# To maintain backward compatibility (if you run this file directly)
if __name__ == "__main__":
    orchestrator = PipelineOrchestrator()
    
    # Simulate the old run() behavior
    df = orchestrator.load_data('interim_csv')
    pipeline = orchestrator.get_pipeline()
    
    print("⚙️ Running Vectorization Pipeline (Fitting on full dataset)...")
    # The fit_transform method will apply all the transformations...
    result = pipeline.fit_transform(df)
    
    print("Pipeline executed successfully. Output shape:", result.shape)
    # print the first 5 rows of the output matrix to verify it worked
    print(result[:5])
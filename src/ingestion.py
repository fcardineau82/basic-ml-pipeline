import pandas as pd
import lxml.etree as ET  # Much faster for large XML
from bs4 import BeautifulSoup
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
import yaml

# --- LOGIC FUNCTIONS ---

def load_xml(config):
    """High-speed XML parsing using lxml with explicit type handling."""
    # Loads the XML data using lxml for efficient parsing, with an option to limit the number of rows for testing. 
    # It also ensures that the grouping column is treated as a string to avoid issues during splitting.
    
    print(f"Loading {config['paths']['raw_xml']}...")
    context = ET.iterparse(config['paths']['raw_xml'], events=('end',), tag='row') 
    # This will iterate through each <row> element in the XML file, which corresponds to a single question, 
    # and extract its attributes into a list of dictionaries. 
    
    rows = [] 
    # We will store the extracted data in a list of dictionaries, which we will later convert into a DataFrame. 
    # This approach is memory efficient and allows us to handle large XML files without loading everything into memory at once.
    
    for i, (event, elem) in enumerate(context):
        # Each <row> element in the XML file contains attributes that correspond to the columns in our dataset.
        rows.append(dict(elem.attrib)) 
        
        # Clear the element from memory to keep our memory usage low.
        elem.clear() 
        
        # Simple mechanism to limit the number of rows we load for testing purposes.
        if config['parsing']['limit_test'] and i >= 500: 
            break
    
    df = pd.DataFrame(rows) 

    # Ensure Group Column is always a string to avoid type-mismatch splits
    group_col = config['splitting']['group_column']
    if group_col in df.columns:
        # Convert to string and handle 'None' and 'nan' as missing values (pd.NA) for consistent processing.
        df[group_col] = df[group_col].astype(str).replace(['None', 'nan'], pd.NA)
        
    return df

def clean_text(df, column="Body"): 
    # This method cleans the specified text column by removing HTML tags using BeautifulSoup. 
    # It uses tqdm to show a progress bar during the cleaning process.
    tqdm.pandas(desc="Cleaning HTML")
    df[column] = df[column].progress_apply(
        lambda x: BeautifulSoup(str(x), "lxml").get_text() if pd.notnull(x) else ""
    )
    return df

def create_quality_label(row):
    """Define what makes a high-quality question with better class balance.
    
    Returns:
        1: High quality (strong positive signals)
        0: Low quality (closed OR negative engagement)
        -1: Uncertain (exclude from training)
    """
    score = row['Score']
    
    # DEFINITE Low Quality: Closed questions
    if pd.notna(row.get('ClosedDate')):
        return 0
    
    # DEFINITE Low Quality: Downvoted (negative score)
    if score < 0:
        return 0
    
    # DEFINITE High Quality: Strong community endorsement
    # Accepted answer AND good score (5+)
    accepted = row.get('AcceptedAnswerId')
    if pd.notna(accepted) and str(accepted).strip() != '' and score >= 5:
        return 1
    
    # DEFINITE High Quality: Very high score even without accepted answer
    if score >= 10:
        return 1
    
    # BORDERLINE Low Quality: Open but zero score (no community interest)
    # These are likely low-quality but we're being conservative
    if score == 0:
        return 0
    
    # UNCERTAIN: Score 1-4 without accepted answer
    # Could be decent but new, or mediocre - exclude these
    return -1

def get_vectorized_inputs_and_label(df, median_threshold=None, use_quality_label=True):
    """Prepares the input features and labels for modeling.
    
    Args:
        df: DataFrame containing the data
        median_threshold: Optional pre-calculated median score threshold (for backwards compatibility)
        use_quality_label: If True, use quality-based labels. If False, use score threshold.
    
    Returns:
        Vector: DataFrame with features (Score column removed)
        Label: Binary Series (1 for high quality, 0 for low quality)
        threshold_or_stats: median_value (for score-based) or None (for quality-based)
    """
    # Ensure Score is numeric
    df["Score"] = pd.to_numeric(df["Score"], errors='coerce')
    
    if use_quality_label:
        print("\n📊 Creating quality-based labels...")
        
        # Create quality labels
        quality_labels = df.apply(create_quality_label, axis=1)
        
        # Filter out uncertain cases (-1)
        valid_mask = (quality_labels != -1)
        print(f"   Total samples: {len(df)}")
        print(f"   Uncertain samples (excluded): {(~valid_mask).sum()}")
        print(f"   Valid samples for training: {valid_mask.sum()}")
        
        # Filter dataframe and labels
        df_filtered = df[valid_mask].copy()
        Label = quality_labels[valid_mask]
        
        # Remove Score column from features
        Vector = df_filtered.drop(columns=['Score'])
        
        print(f"   Label distribution: {Label.value_counts().to_dict()}")
        print(f"   Class balance: {Label.value_counts(normalize=True).to_dict()}")
        
        return Vector, Label, None
    
    else:
        # Original score-based labeling (backwards compatibility)
        if median_threshold is None:
            median_value = df["Score"].median()
        else:
            median_value = median_threshold
        
        Vector = df.drop(columns=['Score'])
        Label = df["Score"] > median_value
        
        return Vector, Label, median_value
 

def split_data(df, config):
    """Handles group-aware data partitioning with Technique 1 (Unique Placeholders)."""
    # This is responsible for performing a group-aware train-test split using the GroupShuffleSplit from scikit-learn.
    # It handles missing values by assigning unique IDs to prevent data leakage.
    
    split_cfg = config['splitting']
    group_col = split_cfg['group_column']

    if group_col not in df.columns:
        raise KeyError(f"Dimension '{group_col}' not found in DataFrame.")

    # Create a deep copy to ensure we don't modify the source pipeline.df
    df = df.copy()
    
    # Handle missing values by giving each row a unique "Anonymous Author" ID
    mask = df[group_col].isna()
    num_missing = mask.sum()
    
    if num_missing > 0:
        print(f"Assigning unique IDs to {num_missing} rows with missing {group_col}...")
        df.loc[mask, group_col] = [f"unknown_{i}" for i in range(num_missing)]

    # Perform Grouped Shuffle
    gss = GroupShuffleSplit(
        n_splits=1, 
        train_size=split_cfg['train_size'], 
        random_state=split_cfg['random_state']
    )
    
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    
    # Return the train and test DataFrames based on the generated indices.
    return df.iloc[train_idx], df.iloc[test_idx]

# --- ORCHESTRATOR ---

def run_ingestion_pipeline(config_path="config.yaml"):
    """Orchestrator for the data ingestion and splitting lifecycle."""
    # This module handles the entire data ingestion lifecycle, including loading the raw XML data, 
    # cleaning the text fields, and performing a group-aware train-test split.
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Parsing
    df = load_xml(config)
    
    # 2. Cleaning
    df = clean_text(df, column="Body")

    # 3. Prepare features and labels (optional, can be used in the next steps of the pipeline)
    Vector, Label, median_value = get_vectorized_inputs_and_label(df)
    print(f"✅ Prepared features and labels. Feature shape: {Vector.shape}, Label distribution: {Label.value_counts().to_dict()}")
    print(f"   Median score threshold: {median_value}")
    print(f"Sample features:\n{Vector.head()}")
    print(f"Sample labels:\n{Label.head()}")
    
    # 4. Splitting
    train_df, test_df = split_data(df, config)

    # Persistence
    train_df.to_csv(config['paths']['train_csv'], index=False)
    test_df.to_csv(config['paths']['test_csv'], index=False)
    
    # to avoid changing interim_csv everywhere in the code, we also save the train_df to the interim_csv path 
    # which is the input for the next steps in the pipeline (clustering and analysis)
    train_df.to_csv(config['paths']['interim_csv'], index=False)
    
    print(f"--- Pipeline Summary ---")
    print(f"Train set saved to: {config['paths']['train_csv']}")
    print(f"Train set size: {len(train_df)} rows")
    print(f"Test set saved to: {config['paths']['test_csv']}")
    print(f"Author overlap check: {set(train_df[config['splitting']['group_column']]).intersection(set(test_df[config['splitting']['group_column']]))}")

if __name__ == "__main__":
    run_ingestion_pipeline()
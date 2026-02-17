from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np # For numerical operations
import re # For regex operations to clean the tags
from tqdm import tqdm
import src.text_analyzer as text_analyzer # To compute the Flesch Reading Ease Score for the quality feature extractor

# This custom transformer extracts features from the 'Title' and 'Body' of questions to help us analyze the quality of the questions. The features include:
# 1. Verbosity (length of title and body)
# 2. Formatting & Effort (whether the title ends with a question mark and the ratio of capital letters in the title)
# 3. Add the Flesch Reading Ease Score to measure readability, which can be an indicator of question quality using the text_analyzer module to compute it based on the number of sentences, words, and syllables in the text.
class QualityFeatureExtractor(BaseEstimator, TransformerMixin):
    # Fit does nothing, just returns self
    def fit(self, X, y=None):
        return self
    # Transform the input DataFrame to extract the new features
    def transform(self, X):
        """
        Expects X to be a DataFrame with columns ['Title', 'Body'].
        Returns a numpy array of shape (n_samples, 5).
        """
        # Create a copy to avoid SettingWithCopy warnings
        df = X.copy()
        
        # Ensure we are working with strings, treating NaNs as empty strings
        df['Title'] = df['Title'].fillna('').astype(str)
        df['Body'] = df['Body'].fillna('').astype(str)
        
        # --- Feature 1: Verbosity (Length) ---
        # Hypothesis: Extremely short questions are lazy; extremely long ones are confusing.
        # Vectorized: .str.len() is much faster than .apply(len)
        df['title_len'] = df['Title'].str.len()
        df['body_len'] = df['Body'].str.len()

        # --- Feature 2: Formatting & Effort ---
        # Hypothesis: Lazy questions often miss question marks or use all-caps.
        df['title_ends_with_qmark'] = df['Title'].str.contains("?", regex=False).astype(int)
        df['body_ends_with_qmark'] = df['Body'].str.contains("?", regex=False).astype(int)
        # Vectorized caps ratio: count uppercase chars / total length
        title_upper_count = df['Title'].str.count(r'[A-Z]')
        title_length = df['Title'].str.len().replace(0, np.nan)
        df['title_caps_ratio'] = (title_upper_count / title_length).fillna(0)

        # --- Feature 3: Readability (Flesch Reading Ease Score) ---
        # Hypothesis: Questions that are too complex or too simple may be of lower quality.
        # Optimised: operate on a combined Series instead of per-row df.apply(axis=1)
        _sentence_split_re = re.compile(r'[.!?]+')
        combined_text = df['Title'] + ' ' + df['Body']

        def _prepare_and_score(text):
            sentences = [s.split() for s in _sentence_split_re.split(text) if s.strip()]
            if not sentences:
                return 0.0
            n_sent = len(sentences)
            n_word = text_analyzer.count_total_number_words(sentences)
            n_syll = text_analyzer.count_total_syllables(sentences)
            if n_sent == 0 or n_word == 0:
                return 0.0
            return text_analyzer.compute_flesch_reading_ease(n_sent, n_word, n_syll)

        # Series.apply is faster than DataFrame.apply(axis=1) — no row-construction overhead
        tqdm.pandas(desc="Computing Flesch scores", unit="row")
        df['flesch_score'] = combined_text.progress_apply(_prepare_and_score)

        # --- Feature 4: Action verb density ---
        # Hypothesis: Questions with action verbs indicate more specific, higher-quality questions.
        action_verbs = ['how to', 'what is', 'why does', 'can i', 'is there',
                        'does anyone', 'help me', 'anyone know',
                        'What', 'How', 'Why', 'should', 'can']
        _action_re = re.compile(
            '|'.join(fr'\b{verb}\b' for verb in action_verbs),
            flags=re.IGNORECASE
        )
        # Vectorized: use .str.findall() and .str.split() on Series (no axis=1 apply)
        match_counts = df['Body'].str.findall(_action_re).str.len()
        word_counts = df['Body'].str.split().str.len().replace(0, np.nan)
        df['action_verbs_density'] = (match_counts / word_counts).fillna(0)

        # Return the dense matrix of new features
        return df[['title_len', 'body_len','title_ends_with_qmark', 'body_ends_with_qmark', 'flesch_score', 'action_verbs_density', 'title_caps_ratio']].values
    
    def get_feature_names_out(self, input_features=None):
            """
            Returns the names of the features produced by the transform method.
            The order must match the order of columns in the returned numpy array.
            """
            return np.array([
                'title_len', 
                'body_len', 
                'title_ends_with_qmark', 
                'body_ends_with_qmark', 
                'flesch_score', 
                'action_verbs_density', 
                'title_caps_ratio'
            ])
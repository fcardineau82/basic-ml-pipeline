"""Unit tests for the ML pipeline components.

Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Tests for text_analyzer.py
# ---------------------------------------------------------------------------
class TestTextAnalyzer:
    """Tests for NLP utility functions in text_analyzer.py."""

    def test_clean_input_strips_and_lowercases(self): # We test that the clean_input function correctly removes leading and trailing whitespace and converts the input string to lowercase. This ensures that the text is standardized before further processing, which can improve the performance of NLP models.
        from src.text_analyzer import clean_input
        assert clean_input("  Hello World  ") == "hello world"

    def test_clean_input_removes_non_ascii(self): # We test that the clean_input function removes non-ASCII characters from the input string. This is important for ensuring that the text is in a consistent format and can be processed by NLP models that may not handle non-ASCII characters well.
        from src.text_analyzer import clean_input
        result = clean_input("café résumé")
        assert all(ord(c) < 128 for c in result)

    def test_preprocess_input_tokenizes(self): # We test that the preprocess_input function correctly tokenizes the input text into sentences and then into words. This is a crucial step in NLP pipelines, as it breaks down the text into manageable pieces for analysis and feature extraction.
        from src.text_analyzer import preprocess_input
        result = preprocess_input("hello world. how are you?")
        assert isinstance(result, list)
        assert len(result) == 2  # Two sentences
        assert all(isinstance(s, list) for s in result)

    def test_count_word_usage(self): # We test that the count_word_usage function accurately counts the occurrences of specified target words in a list of tokens. This is useful for feature engineering, where the frequency of certain words can be an important signal for model training.
        from src.text_analyzer import count_word_usage
        tokens = ["the", "cat", "said", "hello", "said"]
        assert count_word_usage(tokens, ["said"]) == 2
        assert count_word_usage(tokens, ["dog"]) == 0

    def test_compute_flesch_reading_ease(self): # We test that the compute_flesch_reading_ease function calculates the Flesch Reading Ease score correctly based on the known formula. This score is a common readability metric that can be used as a feature in NLP models to assess the complexity of the text.
        from src.text_analyzer import compute_flesch_reading_ease
        # Known formula: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
        score = compute_flesch_reading_ease(
            number_of_sentences=1,
            number_of_words=10,
            number_of_syllables=15
        )
        expected = 206.835 - 1.015 * (10 / 1) - 84.6 * (15 / 10)
        assert abs(score - expected) < 0.01

    def test_flesch_returns_zero_for_empty(self): # We test that the compute_flesch_reading_ease function returns 0.0 when there are no sentences or words, which is a reasonable default to avoid division by zero and indicates that the text is not readable.
        from src.text_analyzer import compute_flesch_reading_ease
        assert compute_flesch_reading_ease(0, 0, 0) == 0.0

    def test_get_word_syllables(self): # We test that the get_word_syllables function correctly retrieves the number of syllables for a given word using the CMU Pronouncing Dictionary. This is important for calculating readability scores and other linguistic features.
        from src.text_analyzer import get_word_syllables
        # "hello" has 2 syllables in CMU dict
        assert get_word_syllables("hello") == 2
        # Punctuation should return 0
        assert get_word_syllables("...") == 0

    def test_unique_words_fraction(self): # We test that the compute_total_unique_words_fraction function correctly calculates the fraction of unique words in a list of tokenized sentences. This is useful for feature engineering, where the diversity of vocabulary can be an important signal for model training.
        from src.text_analyzer import compute_total_unique_words_fraction
        sentences = [["the", "cat", "sat"], ["the", "cat", "ran"]]
        fraction = compute_total_unique_words_fraction(sentences)
        # 4 unique words out of 6 total
        assert abs(fraction - 4 / 6) < 0.01

    def test_unique_words_fraction_empty(self): # We test that the compute_total_unique_words_fraction function returns 0.0 when given an empty list of sentences, which is a reasonable default to avoid division by zero and indicates that there are no unique words.
        from src.text_analyzer import compute_total_unique_words_fraction
        assert compute_total_unique_words_fraction([]) == 0.0


# ---------------------------------------------------------------------------
# Tests for feature_engineering.py
# ---------------------------------------------------------------------------
class TestQualityFeatureExtractor:
    """Tests for the custom QualityFeatureExtractor transformer."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'Title': [
                'How to sort a list in Python?',
                'HELP ME PLZ',
                'What is the best way to handle exceptions in Java?'
            ],
            'Body': [
                'I want to sort a list of integers in ascending order. What is the most efficient way?',
                'code not working',
                'I am trying to handle exceptions properly in my Java application. Can anyone recommend best practices?'
            ]
        })

    def test_output_shape(self, sample_df): # We test that the QualityFeatureExtractor produces an output with the expected shape, which should have the same number of rows as the input DataFrame and a specific number of features (7 in this case). This ensures that the transformer is correctly generating the intended features for each sample.
        from src.feature_engineering import QualityFeatureExtractor
        extractor = QualityFeatureExtractor()
        result = extractor.fit_transform(sample_df)
        assert result.shape == (3, 7)  # 3 samples, 7 features

    def test_output_is_numeric(self, sample_df): # We test that the QualityFeatureExtractor produces numeric output, which is essential for downstream machine learning models that expect numerical input.
        from src.feature_engineering import QualityFeatureExtractor
        extractor = QualityFeatureExtractor()
        result = extractor.fit_transform(sample_df)
        assert np.issubdtype(result.dtype, np.number)

    def test_feature_names(self, sample_df): # We test that the QualityFeatureExtractor provides the correct feature names when calling get_feature_names_out. This is important for interpretability and for ensuring that the features are correctly identified in the output.
        from src.feature_engineering import QualityFeatureExtractor
        extractor = QualityFeatureExtractor()
        extractor.fit(sample_df)
        names = extractor.get_feature_names_out()
        expected = ['title_len', 'body_len', 'title_ends_with_qmark',
                     'body_ends_with_qmark', 'flesch_score', 
                     'action_verbs_density', 'title_caps_ratio']
        assert list(names) == expected

    def test_question_mark_detection(self, sample_df): # We test that the QualityFeatureExtractor correctly identifies whether the title ends with a question mark. This is a specific feature that can be important for distinguishing between questions and statements, which may have different quality characteristics in a Q&A context.
        from src.feature_engineering import QualityFeatureExtractor
        extractor = QualityFeatureExtractor()
        result = extractor.fit_transform(sample_df)
        # Column index 2 = title_ends_with_qmark
        assert result[0, 2] == 1  # "How to sort a list in Python?" has ?
        assert result[1, 2] == 0  # "HELP ME PLZ" has no ?

    def test_handles_nan_values(self): # We test that the QualityFeatureExtractor can handle NaN values in the input DataFrame without throwing errors. This is important for robustness, as real-world data often contains missing values.
        from src.feature_engineering import QualityFeatureExtractor
        df = pd.DataFrame({
            'Title': [None, 'Test?'],
            'Body': ['Some body', None]
        })
        extractor = QualityFeatureExtractor()
        result = extractor.fit_transform(df)
        assert result.shape == (2, 7)
        assert not np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# Tests for ingestion.py
# ---------------------------------------------------------------------------
class TestIngestion:
    """Tests for data ingestion utility functions."""

    def test_create_quality_label_high_quality(self): # We test that the create_quality_label function correctly identifies high-quality questions based on the presence of an accepted answer and a good score, as well as cases where a very high score alone can indicate high quality. This is important for creating a target variable for model training that reflects the quality of the questions.
        from src.ingestion import create_quality_label
        # Score >= 10 triggers high quality even without AcceptedAnswerId
        row = pd.Series({
            'Score': 10,
            'ClosedDate': pd.NA,
            'AcceptedAnswerId': pd.NA
        })
        assert create_quality_label(row) == 1

    def test_create_quality_label_low_quality_closed(self): # We test that the create_quality_label function correctly identifies low-quality questions based on the presence of a ClosedDate, which indicates that the question was closed by the community. This is a strong signal of low quality and should be reflected in the target variable for model training.
        from src.ingestion import create_quality_label
        row = pd.Series({
            'Score': 5,
            'ClosedDate': '2024-01-01',
            'AcceptedAnswerId': pd.NA
        })
        assert create_quality_label(row) == 0

    def test_create_quality_label_low_quality_negative(self): # We test that the create_quality_label function correctly identifies low-quality questions based on a negative score, which indicates that the question has been downvoted by the community. This is another strong signal of low quality and should be reflected in the target variable for model training.
        from src.ingestion import create_quality_label
        row = pd.Series({
            'Score': -2,
            'ClosedDate': pd.NA,
            'AcceptedAnswerId': pd.NA
        })
        assert create_quality_label(row) == 0

    def test_create_quality_label_uncertain(self): # We test that the create_quality_label function correctly identifies uncertain cases where the question does not have strong positive or negative signals. In this case, a question with a moderate score and no accepted answer or closed date should be labeled as -1 (uncertain), which can be excluded from training to improve model performance.
        from src.ingestion import create_quality_label
        row = pd.Series({
            'Score': 3,
            'ClosedDate': pd.NA,
            'AcceptedAnswerId': pd.NA
        })
        assert create_quality_label(row) == -1

    def test_clean_text_removes_html(self): # We test that the clean_text function correctly removes HTML tags from the specified text column. This is important for ensuring that the text data is clean and free of markup, which can interfere with NLP processing and feature extraction.
        from src.ingestion import clean_text
        df = pd.DataFrame({'Body': ['<p>Hello <b>world</b></p>']})
        result = clean_text(df, column='Body')
        assert result['Body'].iloc[0] == 'Hello world'

    def test_clean_text_handles_nan(self): # We test that the clean_text function can handle NaN values in the specified text column without throwing errors. This is important for robustness, as real-world data often contains missing values, and the function should be able to process them gracefully.
        from src.ingestion import clean_text
        df = pd.DataFrame({'Body': [None, '<p>Test</p>']})
        result = clean_text(df, column='Body')
        assert result['Body'].iloc[0] == ''
        assert result['Body'].iloc[1] == 'Test'


# ---------------------------------------------------------------------------
# Tests for model_training.py
# ---------------------------------------------------------------------------
class TestModelTraining:
    """Tests for model factory function."""

    def test_get_model_random_forest(self): # We test that the get_model function correctly constructs a RandomForestClassifier with the specified parameters from the config. 
        # This ensures that our model factory function is properly interpreting the configuration and creating the model with the intended settings, which is crucial for training and performance.
        from src.model_training import get_model
        config = {
            'model': {
                'algorithm': 'random_forest',
                'random_state': 42,
                'random_forest': {
                    'n_estimators': 50,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'class_weight': 'balanced',
                    'oob_score': True
                }
            }
        }
        model = get_model(config)
        assert model.n_estimators == 50
        assert model.max_depth == 10
        assert model.class_weight == 'balanced'
        assert model.oob_score is True

    def test_get_model_logistic_regression(self): # We test that the get_model function correctly constructs a LogisticRegression model when specified in the config. This ensures that our model factory function can handle multiple types of models and is correctly interpreting the configuration to create the appropriate model for training.
        from src.model_training import get_model
        config = {
            'model': {
                'algorithm': 'logistic_regression',
                'random_state': 42
            }
        }
        model = get_model(config)
        from sklearn.linear_model import LogisticRegression
        assert isinstance(model, LogisticRegression)

    def test_get_model_unknown_raises(self): # We test that the get_model function raises a ValueError when an unknown algorithm is specified in the config. This is important for ensuring that our model factory function has proper error handling and provides clear feedback when the configuration is incorrect, which can help prevent silent failures and guide users to fix their config.
        from src.model_training import get_model
        config = {'model': {'algorithm': 'unknown_algo', 'random_state': 42}}
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_model(config)

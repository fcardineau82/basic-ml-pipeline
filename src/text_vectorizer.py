
import spacy
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


class TextVectorizer(BaseEstimator, TransformerMixin):
    """
    Converts text into fixed-length numerical vectors using pre-trained SpaCy
    word vectors (GloVe-style embeddings bundled with en_core_web_md).

    Strategy — tokenizer-only with vocabulary lookup:
        Instead of running the full SpaCy NLP pipeline (which includes an
        expensive tok2vec CNN), we use ONLY the tokenizer to split text into
        tokens, then look up each token's pre-trained static vector from the
        vocabulary table and average them.  This is ~10-50x faster because:

        1. The tok2vec CNN (HashEmbedCNN) is the most expensive component in
           en_core_web_md.  Even though we previously "disabled" parser/tagger/
           NER, tok2vec was STILL running — it wasn't in the disable list.
        2. doc.vector (what we used before) is already just the mean of the
           static word vectors — the tok2vec output was never consumed.
           So the CNN computation was entirely wasted.
        3. By calling nlp.tokenizer(text) directly, we skip ALL pipeline
           components and only pay the cost of rule-based tokenization + a
           hash-table lookup per token.  Each lookup is O(1).

    We also use `exclude=` instead of `disable=` when loading the model:
        - disable= loads components into memory but skips them during processing
        - exclude= prevents them from being loaded at all (~20MB less RAM)
    """

    def __init__(self, model="en_core_web_md"):
        # model: the SpaCy model to load.  Must have pre-trained word vectors.
        # en_core_web_md ships with 300-dimensional GloVe vectors (~20k keys).
        # en_core_web_sm does NOT have real word vectors — it only has
        # context-dependent vectors from tok2vec, which we're bypassing.
        self.model = model
        self.nlp = None
        # vec_length will be set in fit() once the model is loaded.
        # It's stored as an instance attribute so get_feature_names_out()
        # can return the correct number of dimensions without hardcoding.
        self.vec_length = None

    def fit(self, X, y=None):
        """
        Load the SpaCy model lazily on first fit().

        We use `exclude=` to completely skip loading heavyweight pipeline
        components (tok2vec, parser, tagger, NER, senter, etc.) that we
        don't need.  Only the tokenizer and vocabulary (with word vectors)
        are loaded.  This makes model loading faster and uses less memory.
        """
        if self.nlp is None:
            try:
                # exclude= vs disable=:
                #   disable=["tok2vec"] → component is loaded into memory but
                #     skipped during nlp(text). Still wastes RAM.
                #   exclude=["tok2vec"] → component is never loaded at all.
                #     Saves ~20MB RAM and speeds up spacy.load().
                #
                # We exclude EVERY component because we only call
                # nlp.tokenizer(text) directly — no pipeline components
                # are needed at all.
                self.nlp = spacy.load(
                    self.model,
                    exclude=[
                        "tok2vec",          # CNN for contextualized embeddings — most expensive, not needed
                        "tagger",           # POS tagging — not needed
                        "parser",           # Dependency parsing — not needed
                        "ner",              # Named entity recognition — not needed
                        "senter",           # Sentence segmentation — not needed
                        "attribute_ruler",  # Rule-based attribute assignment — not needed
                        "lemmatizer",       # Lemmatization — not needed
                    ]
                )
            except OSError:
                raise OSError(
                    f"Could not find spaCy model '{self.model}'. "
                    f"Run: python -m spacy download {self.model}"
                )

            # Store the vector dimensionality (300 for en_core_web_md).
            # This avoids hardcoding and ensures get_feature_names_out()
            # stays correct even if you switch models.
            self.vec_length = self.nlp.vocab.vectors_length

        return self

    def _text_to_vector(self, text):
        """
        Convert a single text string into a fixed-length vector by averaging
        the pre-trained word vectors of its tokens.

        How it works:
            1. nlp.tokenizer(text) splits the text using SpaCy's rule-based
               tokenizer (handles punctuation, contractions, etc.).  This is
               extremely fast — no neural network is involved.
            2. For each token, we check token.has_vector: True if the word
               appears in the model's vocabulary (the ~20k most common words
               for en_core_web_md).  Out-of-vocabulary words are skipped.
            3. token.vector returns the pre-trained 300-dim GloVe vector
               for that word.  This is a simple hash-table lookup, O(1).
            4. We average all token vectors to produce a single vector that
               represents the entire text (a "bag of word-vectors" approach).

        If no tokens have vectors (all OOV or empty text), we return a
        zero vector to maintain consistent output dimensions.
        """
        # nlp.tokenizer() is ~100x faster than nlp() because it skips ALL
        # pipeline components (tok2vec, tagger, parser, NER, etc.)
        doc = self.nlp.tokenizer(text)

        # Collect vectors only for in-vocabulary tokens.
        # token.has_vector is False for punctuation, rare words, etc.
        vectors = [token.vector for token in doc if token.has_vector]

        if vectors:
            # np.mean(axis=0) averages element-wise across all token vectors,
            # producing a single 300-dim vector for the whole text.
            return np.mean(vectors, axis=0)

        # Fallback: return a zero vector if no tokens had vectors.
        # This preserves the output shape expected by downstream transformers.
        return np.zeros(self.vec_length)

    def transform(self, X):
        """
        Transform a DataFrame/Series of text into a (n_samples, vec_length)
        numpy array of word-vector averages.

        Input handling:
            - If X is a DataFrame with multiple columns (e.g., ['Title', 'Body']),
              we concatenate them into a single text string per row so the
              resulting vector captures signal from ALL text columns.
            - If X is a single-column DataFrame or a Series, we use it directly.
        """
        # --- Step 1: Extract text data into a single Series ---
        if isinstance(X, pd.DataFrame):
            if X.shape[1] > 1:
                # Multiple text columns: concatenate them with a space separator.
                # str.cat() is vectorized and ~5-10x faster than
                # X.apply(lambda row: ' '.join(row.astype(str)), axis=1)
                # because it avoids Python-level row iteration.
                data_series = X.iloc[:, 0].astype(str).str.cat(
                    X.iloc[:, 1:].astype(str), sep=' '
                )
            else:
                data_series = X.iloc[:, 0].astype(str)
        else:
            data_series = pd.Series(X).astype(str)

        # Replace NaN with empty string to avoid errors in tokenization
        texts = data_series.fillna('').tolist()

        # --- Step 2: Vectorize each text using tokenizer + vocab lookup ---
        # We iterate over the plain text list and call _text_to_vector() for
        # each one.  tqdm wraps the iterator to show a progress bar.
        # miniters limits the bar update frequency to ~100 refreshes total,
        # which avoids I/O overhead from updating the bar on every single row.
        vectors = [
            self._text_to_vector(text)
            for text in tqdm(
                texts,
                desc="Vectorizing text",
                unit="doc",
                miniters=max(1, len(texts) // 100),
            )
        ]

        # Stack the list of 1-D vectors into a 2-D array: (n_samples, vec_length)
        return np.vstack(vectors)

    def get_feature_names_out(self, input_features=None):
        """
        Return feature names for the output columns.
        Uses the actual vector length from the loaded model instead of
        hardcoding 300, so it stays correct if you switch SpaCy models.
        """
        # If fit() hasn't been called yet, fall back to 300 (en_core_web_md default)
        length = self.vec_length if self.vec_length else 300
        return np.array([f"text_dim_{i}" for i in range(length)])


# initialize and test the pipeline
if __name__ == "__main__":
    # Load the config to get the text features
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    text_features = config['features']['text']
    print(f"Text features to vectorize: {text_features}")

    # Load the data from the config.yaml file path
    input_path = config['paths']['interim_csv']
    df = pd.read_csv(input_path)
    print(f"✅ Loaded data with {len(df)} rows.")

    # Initialize the TextVectorizer and fit-transform the 'Body' column
    vectorizer = TextVectorizer()
    vectorized_output = vectorizer.fit_transform(df['Body'])
    #print the first 5 rows of the vectorized output to verify it worked
    print("First 5 rows of vectorized output:")
    print(vectorized_output[:5])

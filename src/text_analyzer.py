import argparse
import functools
import unicodedata
import nltk
from nltk.corpus import cmudict


def _ensure_nltk_data():
    """Download required NLTK data if not already present."""
    for resource in ['punkt_tab', 'cmudict']:
        try:
            nltk.data.find(f'corpora/{resource}' if resource == 'cmudict' else f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()
DICTIONARY = cmudict.dict()




def parse_arguments():
    """
    :return: The text to be edited
    """

    parser = argparse.ArgumentParser(description="Receive text to be edited.")
    parser.add_argument(
        'text',
        metavar='input text',
        type=str,
    )

    args = parser.parse_args()
    return args.text  

def clean_input(text):
    """
    Cleans the input text by removing leading/trailing whitespace and converting to lowercase.

    :param text: The text to be cleaned
    :return: The cleaned text without non-ASCII characters removed
    """
    
    # normalize the text
    text = text.strip().lower()
    text = unicodedata.normalize('NFKD', text)

    # remove non-ASCII characters
    text = ''.join(c for c in text if ord(c) < 128)

    return str(text)

def preprocess_input(text):
    """

    :param text: Sanitized text
    :return: Text to be ready to be fed to analysis, by having sentences and words tokenized
    """
    
    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text)

    # Tokenize each sentence into words
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

    return tokenized_sentences

def count_word_usage(token_list, target_words):
    """
    Count the occurrences of target words in a list of tokens.

    :param token_list: List of word tokens
    :param target_words: List of target words to count
    :return: Count of target words in the token list
    """
    count = 0
    for token in token_list:
        if token in target_words:
            count += 1
    return count

def compute_total_unique_words_fraction(sentence_list):
    """
    Compute the fraction of unique words in the entire text.

    :param sentence_list: List of sentences , each being a list of words
    :return: Fraction of unique words in the text
    """
    all_words = [word for sentence in sentence_list for word in sentence]
    unique_words = set(all_words)
    if len(all_words) == 0:
        return 0.0
    return len(unique_words) / len(all_words)


@functools.lru_cache(maxsize=None)
def get_word_syllables(word):
    """
    Get the number of syllables in a word.
    Cached with lru_cache to avoid repeated dictionary lookups for common words.

    :param word: The word to analyze
    :return: Number of syllables in the word
    """
    word = word.lower()
    if word in DICTIONARY:
        # Count vowel-sound markers (digits) in the first pronunciation variant
        return sum(1 for ph in DICTIONARY[word][0] if ph[-1].isdigit())
    elif not word.isalpha():
        return 0
    else:
        return 1  # Fallback for words not found in cmudict


def count_total_syllables(sentence_list):
    """
    Count the total number of syllables in the text.
    """
    total_syllables = 0
    for sentence in sentence_list:
        for word in sentence:
            total_syllables += get_word_syllables(word) 
    return total_syllables

def count_total_number_words(sentence_list):
    """
    Count the total number of words in the text.

    :param sentence_list: List of sentences , each being a list of words
    :return: Total number of words in the text
    """
    total_words = 0
    for sentence in sentence_list:
        for word in sentence:
            if word.isalpha():  # Check if the token is a word (contains only alphabetic characters)
                total_words += 1

    return total_words


def compute_flesch_reading_ease(number_of_sentences, number_of_words, number_of_syllables):
    """
    Compute the Flesch Reading Ease score.

    :param number_of_sentences: Total number of sentences
    :param number_of_words: Total number of words
    :param number_of_syllables: Total number of syllables
    :return: Flesch Reading Ease score
    """
    if number_of_sentences == 0 or number_of_words == 0:
        return 0.0
    # Flesch Reading Ease formula
    flesch_score = 206.835 - 1.015 * (number_of_words / number_of_sentences) - 84.6 * (number_of_syllables / number_of_words)
    return flesch_score

def get_suggestions(sentence_list):
    """
    Return a string containing our suggestion

    :param sentence_list: List of sentences , each being a list of words
    :return: Suggestions to improve the input text
    """
    told_said_usage = sum(
        (count_word_usage(tokens, ["told", "said"]) for tokens in sentence_list)
    )

    but_and_usage = sum(
        (count_word_usage(tokens, ["but", "and"]) for tokens in sentence_list)
    )

    wh_adverb_usage = sum(
        (count_word_usage(
            tokens, 
            [
                "when",
                "where", 
                "why", 
                "how",
                "whence",
                "whereby",
                "whereupon",
            ],
        )
        for tokens in sentence_list)
    )
    # Build the result string
    result_str = ""
    # Compile count for usage of adverbs
    adverb_usage = f"Adverb usage:\n- 'told'/'said' usage: {told_said_usage}\n- 'but'/'and' usage: {but_and_usage}\n- WH-adverb usage: {wh_adverb_usage}\n"
    # append to result string the adverb count
    result_str += adverb_usage
    # Calculate average word length
    average_word_length = sum(
        (len(word) for sentence in sentence_list for word in sentence)
    ) / sum((len(sentence) for sentence in sentence_list)) if sum((len(sentence) for sentence in sentence_list)) > 0 else 0 
    result_str += f"Average word length: {average_word_length:.2f}\n"
    # Calculate unique word fraction
    unique_word_fraction = compute_total_unique_words_fraction(sentence_list)

    result_str += f"Unique word fraction: {unique_word_fraction:.2f}\n"
    # Format the word statistics
    word_stats = " Average word length %.2f, fraction of unique words %.2f" %(
        average_word_length,
        unique_word_fraction,
    )

    #Calculate the number of syllables
    number_of_syllables = count_total_syllables(sentence_list)
    result_str += f"Total number of syllables: {number_of_syllables}\n"

    #Calculate the number of words
    number_of_words = count_total_number_words(sentence_list)
    result_str += f"Total number of words: {number_of_words}\n"

    #Calculate the number of sentences
    number_of_sentences = len(sentence_list)
    result_str += f"Total number of sentences: {number_of_sentences}\n"

    #calculate a flesch reading ease score
    flesch_score = compute_flesch_reading_ease(number_of_sentences, number_of_words, number_of_syllables)
    result_str += f"Flesch Reading Ease Score: {flesch_score:.2f}\n"

    return result_str


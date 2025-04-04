from sklearn.metrics import classification_report
from nltk.corpus import words
from preprocess import load_conll_data
from pathlib import Path
from lingua import Language, LanguageDetectorBuilder
import emoji
import spacy
import re
import nltk
nltk.download("words")


nlp_en = spacy.load("en_core_web_sm")
nlp_sp = spacy.load("es_core_news_sm")


def word_is_ne(word):
    """
    Checks if a word is a named entity
    (PERSON, ORG, or GPE) in English and Spanish
    Args:
        word (str): input word to check
    Returns:
        label (str): 'ne' if the word is a named entity, 'unk' otherwise
    """

    doc_en = nlp_en(word)
    if doc_en[0].ent_type_:
        if doc_en[0].ent_type_ in {"PERSON", "ORG", "GPE"}:
            return "ne"

    doc_sp = nlp_sp(word)
    if doc_sp[0].ent_type_:
        if doc_sp[0].ent_type_ in {"PERSON", "ORG", "GPE"}:
            return "ne"
    return "unk"


def find_emoji(word):
    """
    Checks if a word is an emoji
    Args:
        word (str): input word to check
    Returns:
        label (str): 'other' if the word is an emoji, 'unk' otherwise
    """
    if emoji.emoji_count(word) > 0:
        return "other"
    return "unk"


def find_symbols(word):
    """
    Checks if a word contains symbols
    Args:
        word (str): input word to check
    Returns:
        label (str): 'other' if the word contains symbols, 'unk' otherwise
    """
    pattern = r'[\d!@#$%^&*()_+\-=\[\]{};":\\|,.<>/?]+'
    if re.search(pattern, word):
        return "other"
    return "unk"


def word_is_language(word):
    """
    Checks if a word is English, Spanish, or mixed using Lingua
    Args:
        word (str): input word to check
    Returns:
        label (str):
            'lang1' if the word is English,
            'lang2' if the word is Spanish,
            'mixed' if the word is a mix of English and Spanish
    """
    word = word.lower()

    languages = [Language.ENGLISH, Language.SPANISH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    # extract all the confidence values
    confidence_values = detector.compute_language_confidence_values(word)

    # initialize confidence scores for english and spanish
    en_confidence = 0.0
    es_confidence = 0.0

    # extract the confidence scores for English and Spanish
    # and assign to the variables
    for cv in confidence_values:
        if cv.language == Language.ENGLISH:
            en_confidence = cv.value
        elif cv.language == Language.SPANISH:
            es_confidence = cv.value

    # initializing thresholds
    # the confidence_threshold is the minimum for a word to be considered part of a language
    # the mixed_threshold defines the ambiguous range where text is considered mixed
    # mixed words: both en_confidence and es_confidence ≥ 0.3 (confidence_threshold)
    # and neither en_confidence nor es_confidence ≥ 0.6 (1-mixed_threshold)
    confidence_threshold = 0.3
    mixed_threshold = 0.4

    if len(word) >= 5 and en_confidence > mixed_threshold and es_confidence > mixed_threshold:
        return "mixed"
    elif en_confidence > confidence_threshold:
        return "lang1"
    elif es_confidence > confidence_threshold:
        return "lang2"
    else:
        return "unk"


def classify_words(word):
    """
    Classifies a word into categories (emoji, symbol, named entity,
    or language).

    The function checks the word in this priority order:
    1. Emoji detection
    2. Symbol detection
    3. Named Entity recognition (PERSON/ORG/GPE)
    4. Language detection (English/Spanish/mixed)
    If none match, returns 'unk' (unknown).
    Args:
        word (str): input word to check
    Returns:
        label (str):
            - 'other' (if the word is an emoji)
            - 'other' (if the word contains any punctuation)
            - 'ne' (if the word is a named entity)
            - 'lang1' (English word)
            - 'lang2' (Spanish word)
            - 'mixed' (mixed word)
            - 'unk' (unknown/unclassified)
    """
    label = "unk"

    if label == "unk":
        label = find_emoji(word)
        if label != "unk":
            return label

    if label == "unk":
        label = find_symbols(word)
        if label != "unk":
            return label

    if label == "unk":
        label = word_is_ne(word)
        if label != "unk":
            return label

    if label == "unk":
        label = word_is_language(word)
        if label == "lang1":
            return "lang1"
        if label == 'lang2':
            return 'lang2'

    return label


def add_baseline_predictions(input_file: Path, output_file: Path):
    """
    Generates baseline predictions for CONLL data and writes results
    Processes input file to:
    1. Write predictions to output_file
    2. Print classification report

    Args:
        input_file: Path to input conll file
        output_file: Path to write output
    """
    sentences, true_labels = load_conll_data(input_file)

    all_true_labels = []
    all_predicted_labels = []
    with output_file.open('w', encoding='utf-8') as f_out:
        for i, (sentence_tokens, sent_true_labels) in enumerate(zip(sentences, true_labels)):

            f_out.write(f"# sent_enum = {i + 1}\n")

            for token, true_label in zip(sentence_tokens, sent_true_labels):
                predicted_label = classify_words(token)
                f_out.write(f"{token}\t{true_label}\t{predicted_label}\n")
                all_true_labels.append(true_label)
                all_predicted_labels.append(predicted_label)

            f_out.write("\n")

    # Classification report
    print("Classification Report (Baseline):")
    print(classification_report(all_true_labels, all_predicted_labels))


input_file = Path("data/dev.conll")
output_file = Path("data/dev_baseline.conll")
add_baseline_predictions(input_file, output_file)

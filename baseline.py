from sklearn.metrics import classification_report
from nltk.corpus import words
from preprocess import load_conll_data
from pathlib import Path
from lingua import Language, LanguageDetectorBuilder
import emoji
from preprocess import *
import spacy
import re
import nltk
nltk.download("words")


nlp_en = spacy.load("en_core_web_sm")
nlp_sp = spacy.load("es_core_news_sm")


def word_is_ne(word):
    """ Check if the word is a named entity """

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
    """ Check if the word is an emoji"""
    if emoji.emoji_count(word) > 0:
        return "other"
    return "unk"


def find_symbols(word):
    """ Check if the word contains symbols"""
    pattern = r'[\d!@#$%^&*()_+\-=\[\]{};":\\|,.<>/?]+'
    if re.search(pattern, word):
        return "other"
    return "unk"


def word_is_language(word):
    """Check if the word is English or Spanish using Lingua"""
    word = word.lower()

    languages = [Language.ENGLISH, Language.SPANISH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    language = detector.detect_language_of(word)

    if language == Language.ENGLISH:
        return "lang1"

    elif language == Language.SPANISH:
        return "lang2"

    return "unk"


def classify_words(word):
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
    sentences, true_labels = load_conll_data(input_file)

    all_true_labels = []
    all_predicted_labels = []
    with output_file.open('w', encoding='utf-8') as f_out:
        for sent_idx, (sentence_tokens, sent_true_labels) in enumerate(zip(sentences, true_labels)):

            f_out.write(f"# sent_enum = {sent_idx + 1}\n")

            for token, true_label in zip(sentence_tokens, sent_true_labels):
                predicted_label = classify_words(token)
                f_out.write(f"{token}\t{true_label}\t{predicted_label}\n")

                all_true_labels.append(true_label)
                all_predicted_labels.append(predicted_label)

            f_out.write("\n")
    print("Classification Report (Baseline):")
    print(classification_report(all_true_labels, all_predicted_labels))


input_file = Path("data/dev.conll")
output_file = Path("data/dev_baseline.conll")
add_baseline_predictions(input_file, output_file)

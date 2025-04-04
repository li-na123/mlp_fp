from nltk.corpus import words
from preprocess_baseline import preprocess_test_conll

from pathlib import Path
from lingua import Language, LanguageDetectorBuilder
import emoji
from preprocess import *
import spacy
# import urbandictionary as ud
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


input_file = Path("data/test.conll")
output_file = Path("data/test_baseline.conll")

test_sentences, test_labels = preprocess_test_conll(input_file, output_file)

for i in range(25):
    print(f"SENTENCE {i}: {test_sentences[i]}")

    updated_labels = []
    for word in test_sentences[i]:
        updated_labels.append(classify_words(word))

    test_labels[i] = updated_labels

    print(f"UPDATED LABELS {i}: {test_labels[i]}")

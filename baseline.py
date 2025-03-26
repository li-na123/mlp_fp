import emoji
import py3langid as langid
from preprocess import *
from langdetect import detect_langs


def find_emoji():
    pass

train_sentences, train_labels = load_conll_data("data/train.conll")

for i in range(25):
    print(f"SENTENCE {i}:", train_sentences[i])
    print(f"LABELS   {i}:", train_labels[i])
    print(detect_langs(train_sentences[i]))
    



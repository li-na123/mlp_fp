from preprocess import load_conll_data

train_sentences, train_labels = load_conll_data("data/train.conll")

# sample
for i in range(25):
    print(f"SENTENCE {i}:", train_sentences[i])
    print(f"LABELS   {i}:", train_labels[i])

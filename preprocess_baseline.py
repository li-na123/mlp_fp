from pathlib import Path


def preprocess_test_conll(input_filepath: Path, output_filepath: Path):
    sentences_words = []
    sentences_labels = []

    with input_filepath.open("r", encoding="utf-8") as f, output_filepath.open("w", encoding="utf-8") as out_f:
        sentence_words = []
        sentence_labels = []

        for line in f:
            line = line.strip()

            if not line:
                if sentence_words:

                    sentences_words.append(sentence_words)
                    sentences_labels.append(sentence_labels)
                    sentence_words = []
                    sentence_labels = []
                out_f.write("\n")
                continue

            word = line.strip()
            sentence_words.append(word)
            sentence_labels.append('unk')
            out_f.write(f"{word}\tunk\n")

        if sentence_words:
            sentences_words.append(sentence_words)
            sentences_labels.append(sentence_labels)

    return sentences_words, sentences_labels

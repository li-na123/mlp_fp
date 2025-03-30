from pathlib import Path

def load_conll_data(filepath: Path):
    sentences = []
    labels = []
    current_tokens = []
    current_labels = []

    with filepath.open("r", encoding="utf-8") as f:

        for line in f:
            line = line.strip()

            if line.startswith("# sent_enum"):
                if current_tokens:  # Save previous sentence
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
                continue

            if not line:
                continue  # skip empty lines

            parts = line.split()
            if len(parts) != 2:
                continue  # Skip malformed lines

            token, label = parts

            # Normalize labels
            if label.lower() in {"fw", "ambiguous"}:
                label = "unk"

            current_tokens.append(token)
            current_labels.append(label.lower())

        # Don't forget to add the last sentence
        if current_tokens:
            sentences.append(" ".join(current_tokens))
            labels.append(current_labels)

    return sentences, labels

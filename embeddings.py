from pathlib import Path
import numpy as np
import torch


def tokenize_and_align(tokens, labels, tokenizer):
    """
    Tokenizes a list of tokens using BERT (with is_split_into_words=True) and aligns the original
    labels to the subword tokens. Special tokens are assigned a placeholder ("PAD").
    Returns the tokenizer encoding and the aligned label list.
    """
    encoding = tokenizer(
        [tokens],
        is_split_into_words=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    # Get the mapping from subword to original token index
    word_ids = encoding.word_ids(batch_index=0)
    aligned_labels = []
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append("PAD") # Special tokens [CLS] and [SEP]
        else:
            aligned_labels.append(labels[word_idx])
    return encoding, aligned_labels

def extract_weighted_embeddings(encoding, model, device):
    """
    Performs a forward pass through the model to extract embeddings.
    Uses the weighted sum of the last four hidden layers.
    Returns a numpy array of shape (seq_length, hidden_size).
    """
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    # outputs.hidden_states is a tuple with one tensor per layer (including embedding layer)
    # Get the last four hidden states and compute their average
    # (This is generally more effective than just grabbing the last hidden layer)
    hidden_states = outputs.hidden_states  # tuple of (layer_count, batch_size, seq_length, hidden_size)
    last_four = torch.stack(hidden_states[-4:], dim=0)  # shape: (4, 1, seq_length, hidden_size)
    combined = torch.mean(last_four, dim=0)  # shape: (1, seq_length, hidden_size)
    embeddings = combined.squeeze(0).cpu().numpy()  # shape: (seq_length, hidden_size)
    return embeddings

def process_sample(tokens, labels, tokenizer, model, device):
    """
    Processes a single sample by tokenizing, aligning labels, and extracting embeddings.
    Returns the embeddings and aligned labels.
    """
    encoding, aligned_labels = tokenize_and_align(tokens, labels, tokenizer)
    embeddings = extract_weighted_embeddings(encoding, model, device)
    return embeddings, aligned_labels

def save_sample(embeddings, aligned_labels, output_dir: Path, sample_id: int):
    """
    Saves the sample's embeddings and aligned labels into a compressed .npz file.
    """
    filename = output_dir / f"sample_{sample_id:06d}.npz"   
    np.savez_compressed(filename, embeddings=embeddings, labels=aligned_labels)

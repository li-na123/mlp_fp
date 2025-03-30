import random
from pathlib import Path

from transformers import BertTokenizerFast, BertModel
from tqdm import tqdm
import torch

from preprocess import load_conll_data
from embeddings import process_sample, save_sample


def main():
    conll_filepath = Path("data/train.conll")
    train_output_dir = Path("embeddings") / "train"
    dev_output_dir = Path("embeddings") / "dev"
    train_ratio = 0.9  # 90% train, 10% dev
    random_seed = 36

    # Create output directories
    train_output_dir.mkdir(parents=True, exist_ok=True)
    dev_output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(random_seed)
    
    # Load data
    sentences, all_labels = load_conll_data(conll_filepath)
    print(f"Loaded {len(sentences)} sentences.")
    
    # Initialize tokenizer and model (cased multilingual BERT)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    # We're not updateing the weights of the BERT model so we put in in evaluation mode
    model.eval()
    # If your machines have cuda (nvida gpu) set up on your machine it will speed up getting the embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    sample_counter = 0
    for tokens, labels in tqdm(zip(sentences, all_labels), total=len(sentences), desc="Processing samples"):
        embeddings, aligned_labels = process_sample(tokens, labels, tokenizer, model, device)
        # Randomly assign the sample to train or dev
        output_dir = train_output_dir if random.random() < train_ratio else dev_output_dir
        sample_counter += 1
        save_sample(embeddings, aligned_labels, output_dir, sample_counter)
    
    print(f"Processed and saved {sample_counter} samples in '{train_output_dir}' and '{dev_output_dir}'.")

if __name__ == "__main__":
    main()


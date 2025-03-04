from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# Create vocabulary for input and output sequence
def build_vocab(texts: pd.Series, min_freq: int = 2) -> dict[str, int]:
    counter: Counter = Counter()
    for text in texts:
        counter.update(text)
    vocab: dict[str, int] = {
        "<pad>": 0,
        "<unk>": 1,
        "<sos>": 2,
        "<eos>": 3,
    }
    for char, count in counter.items():
        if count >= min_freq:
            vocab[char] = len(vocab)
    return vocab


# Tokenization function
def tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(char, vocab["<unk>"]) for char in text]


# Function to pad sequences to the same length
def pad_sequence(
    sequences: list[list[int]], max_len: int, padding_value: int = 0
) -> list[list[int]]:
    padded_sequences: list[list[int]] = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = seq + [padding_value] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len]
        padded_sequences.append(padded_seq)
    return padded_sequences


# Plotting training/validation loss
def plot_result(train_losses: list[float], valid_losses: list[float]) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, "b-", label="Training Loss")
    plt.plot(
        range(1, len(valid_losses) + 1), valid_losses, "r-", label="Validation Loss"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.close()


# prepare the data
def prepare_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load data and create train/test split
    df: pd.DataFrame = pd.read_csv(path)

    # Create id column if it doesn't exist (combining sentence_id and token_id)
    if (
        "id" not in df.columns
        and "sentence_id" in df.columns
        and "token_id" in df.columns
    ):
        df["id"] = df["sentence_id"].astype(str) + "_" + df["token_id"].astype(str)
    # If no id column exists at all, create a simple index as id
    elif "id" not in df.columns:
        df["id"] = df.index.astype(str)

    return train_test_split(df, test_size=0.2, random_state=42)

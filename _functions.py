import pandas as pd
from matplotlib import pyplot as plt


# Create vocabulary for input and output sequence
def build_vocab(texts: pd.Series) -> dict[str, int]:
    vocab: dict[str, int] = {
        "<pad>": 0,
        "<unk>": 1,
        "<sos>": 2,
        "<eos>": 3,
    }
    for text in texts:
        for c in text:
            if c not in vocab:
                vocab[c] = len(vocab)
    return vocab


# Tokenization function
def tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(c, vocab["<unk>"]) for c in text]


# Function to pad sequences to the same length
def pad_sequence(
    sequences: list[list[int]], max_len: int, padding_value: int
) -> list[list[int]]:
    # Output sequences
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


# add sentence_id and token_id column to id column
def convert_df_id(df: pd.DataFrame) -> pd.DataFrame:
    # create id column
    col: pd.Series = df["sentence_id"].astype(str) + "_" + df["token_id"].astype(str)
    # drop sentence_id and token_id
    df = df.drop(columns=["sentence_id", "token_id"])
    # assign id column to the "new" data frame
    df["id"] = col

    return df

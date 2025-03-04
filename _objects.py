import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from _functions import pad_sequence, tokenize


# Create dataset class
class TextNormalizationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        input_vocab: dict[str, int],
        output_vocab: dict[str, int],
        max_input_len: int = 50,
        max_output_len: int = 50,
    ):
        self.df: pd.DataFrame = df
        self.input_vocab: dict[str, int] = input_vocab
        self.output_vocab: dict[str, int] = output_vocab
        self.max_input_len: int = max_input_len
        self.max_output_len: int = max_output_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_text = self.df.iloc[idx]["before"]

        # For training data, we have 'after' column
        if "after" in self.df.columns:
            output_text = self.df.iloc[idx]["after"]
        else:
            output_text = ""  # For test data

        # Tokenize input and output
        input_tokens: list[int] = tokenize(input_text, self.input_vocab)
        input_tokens = input_tokens[: self.max_input_len - 1] + [
            self.input_vocab["<eos>"]
        ]
        input_tokens = pad_sequence(
            [input_tokens], self.max_input_len, self.input_vocab["<pad>"]
        )[0]

        if output_text:
            output_tokens = [self.output_vocab["<sos>"]] + tokenize(
                output_text, self.output_vocab
            )
            output_tokens = output_tokens[: self.max_output_len - 1] + [
                self.output_vocab["<eos>"]
            ]
            output_tokens = pad_sequence(
                [output_tokens], self.max_output_len, self.output_vocab["<pad>"]
            )[0]
        else:
            output_tokens = [self.output_vocab["<pad>"]] * self.max_output_len

        # Always ensure we have an id field
        if "id" in self.df.columns:
            id_val = self.df.iloc[idx]["id"]
        else:
            id_val = str(idx)

        return {
            "input": torch.tensor(input_tokens, dtype=torch.long),
            "output": torch.tensor(output_tokens, dtype=torch.long),
            "id": id_val,
        }


# Create the encoder-decoder model
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding: nn.Embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn: nn.GRU = nn.GRU(
            emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, seq_len, emb_dim]

        outputs, hidden = self.rnn(embedded)
        # outputs: [batch_size, seq_len, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]

        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input: [batch_size, 1]
        # hidden: [n_layers, batch_size, hid_dim]

        input = input.unsqueeze(1)  # [batch_size, 1]

        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, emb_dim]

        output, hidden = self.rnn(embedded, hidden)
        # output: [batch_size, 1, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]

        prediction = self.fc_out(output.squeeze(1))
        # prediction: [batch_size, output_dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.device: torch.device = device

    def forward(self, src, trg, teacher_forcing_ratio: float = 0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Last hidden state of the encoder is used as the initial hidden state of the decoder
        _, hidden = self.encoder(src)

        # First input to the decoder is the <sos> tokens
        input = trg[:, 0]

        for t in range(1, trg_len):
            # Insert input token embedding, previous hidden state
            output, hidden = self.decoder(input, hidden)

            # Place predictions in a tensor holding predictions for each token
            outputs[:, t] = output

            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            # Get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # If teacher forcing, use actual next token as next input, else use predicted token
            input = trg[:, t] if teacher_force else top1

        return outputs

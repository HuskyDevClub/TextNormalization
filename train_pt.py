import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from _functions import build_vocab, plot_result, prepare_data, train_test_split
from _objects import Decoder, Encoder, Seq2Seq, TextNormalizationDataset

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda")

# Load data as train/test split
train_df, test_df = prepare_data("en_train.csv")


# Build vocabularies for input and output
input_vocab = build_vocab(train_df["before"])
output_vocab = build_vocab(train_df["after"])

# Inverse vocabularies for decoding
inverse_input_vocab = {v: k for k, v in input_vocab.items()}
inverse_output_vocab = {v: k for k, v in output_vocab.items()}


# Define model hyperparameters
INPUT_DIM = len(input_vocab)
OUTPUT_DIM = len(output_vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# Create the encoder and decoder
encoder: Encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
decoder: Decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

# Create the model
model = Seq2Seq(encoder, decoder, device).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=output_vocab["<pad>"])
optimizer = optim.Adam(model.parameters())


# Function to train the model
def train(model, dataloader, optimizer, criterion, clip=1):
    model.train()
    epoch_loss = 0

    # Create a progress bar
    total_batches = len(dataloader)
    print(f"\nTraining: 0/{total_batches} batches processed", end="\r")

    for i, batch in enumerate(dataloader):
        src = batch["input"].to(device)
        trg = batch["output"].to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        # trg: [batch_size, trg_len]
        # output: [batch_size, trg_len, output_dim]

        output_dim = output.shape[-1]

        # Exclude the first token (<sos>)
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        # Calculate loss
        loss = criterion(output, trg)

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update parameters
        optimizer.step()

        epoch_loss += loss.item()

        # Update progress bar every 10 batches
        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(
                f"Training: {i+1}/{total_batches} batches processed | Current loss: {loss.item():.4f}",
                end="\r",
            )

    avg_loss = epoch_loss / len(dataloader)
    print(f"\nTraining completed. Average batch loss: {avg_loss:.4f}" + " " * 30)
    return avg_loss


# Function to evaluate the model
def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0

    # Create a progress bar
    total_batches = len(dataloader)
    print(f"Evaluating: 0/{total_batches} batches processed", end="\r")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src = batch["input"].to(device)
            trg = batch["output"].to(device)

            output = model(src, trg, 0)  # Turn off teacher forcing

            # trg: [batch_size, trg_len]
            # output: [batch_size, trg_len, output_dim]

            output_dim = output.shape[-1]

            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

            # Update progress bar periodically
            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                print(f"Evaluating: {i+1}/{total_batches} batches processed", end="\r")

    avg_loss = epoch_loss / len(dataloader)
    print(f"\nEvaluation completed. Average batch loss: {avg_loss:.4f}" + " " * 30)
    return avg_loss


# Function to predict on test data
def predict(model, dataloader, inverse_output_vocab):
    model.eval()
    predictions = []
    ids = []

    # Create a progress bar
    total_batches = len(dataloader)
    print(f"Predicting: 0/{total_batches} batches processed", end="\r")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src = batch["input"].to(device)
            id_batch = batch["id"]

            batch_size = src.shape[0]

            # Encoder outputs
            _, hidden = model.encoder(src)

            # First input to the decoder is the <sos> token
            input = torch.tensor([output_vocab["<sos>"]] * batch_size).to(device)

            # Store predictions for each sequence in the batch
            batch_outputs = [[] for _ in range(batch_size)]

            for _ in range(50):  # Adjust max length as needed
                output, hidden = model.decoder(input, hidden)
                pred_tokens = output.argmax(1)

                # Add predicted tokens to each sequence's output
                for i, token in enumerate(pred_tokens):
                    batch_outputs[i].append(token.item())

                # Update input for next time step
                input = pred_tokens

            # Convert token indices to characters for each sequence
            for i in range(batch_size):
                tokens = batch_outputs[i]
                # Find where the sequence ends (at <eos> token)
                if output_vocab["<eos>"] in tokens:
                    end_idx = tokens.index(output_vocab["<eos>"])
                    tokens = tokens[:end_idx]

                # Convert tokens to text
                text = "".join(
                    [
                        inverse_output_vocab.get(token, "")
                        for token in tokens
                        if token
                        not in [
                            output_vocab["<sos>"],
                            output_vocab["<eos>"],
                            output_vocab["<pad>"],
                        ]
                    ]
                )

                predictions.append(text)
                ids.append(id_batch[i])

            # Update progress bar periodically
            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                print(f"Predicting: {i+1}/{total_batches} batches processed", end="\r")

    print(
        f"\nPrediction completed. Generated {len(predictions)} predictions." + " " * 30
    )
    return ids, predictions


# Prepare the data
train_dataset = TextNormalizationDataset(train_df, input_vocab, output_vocab)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create a validation set from the training data
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=43)
val_dataset = TextNormalizationDataset(val_df, input_vocab, output_vocab)
val_loader = DataLoader(val_dataset, batch_size=64)

# Testing dataset
test_dataset = TextNormalizationDataset(test_df, input_vocab, output_vocab)
test_loader = DataLoader(test_dataset, batch_size=64)

# Training loop
N_EPOCHS: int = 20
# how many epochs for early stopping, -1 to disable, 0 to stop immediately
early_stop_patience: int = 5
# best validation lost
best_valid_loss = float("inf")

print(f"Starting training for {N_EPOCHS} epochs...")
print(f"Training on {len(train_dataset)} examples")
print(f"Validating on {len(val_dataset)} examples")
print(f"Testing on {len(test_dataset)} examples")

# For tracking metrics
train_losses: list[float] = []
valid_losses: list[float] = []

for epoch in range(N_EPOCHS):

    print(f"\n{'='*50}")
    print(f"Epoch: {epoch+1}/{N_EPOCHS}")

    train_loss: float = train(model, train_loader, optimizer, criterion)
    valid_loss: float = evaluate(model, val_loader, criterion)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {valid_loss:.4f}")

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "best-model.pt")
        print(f"New best model saved with validation loss: {valid_loss:.4f}")

    # Early stopping check
    if (
        early_stop_patience >= 0
        and epoch >= early_stop_patience
        and valid_losses[-1] > valid_losses[-2] > valid_losses[-3]
    ):
        print(
            f"Validation loss increasing for {early_stop_patience} consecutive epochs. Early stopping..."
        )
        break

print(f"\n{'='*50}")
print("Training complete!")
print(f"Best validation loss: {best_valid_loss:.4f}")

# Print training summary
print("\nTraining Summary:")
for epoch, (t_loss, v_loss) in enumerate(zip(train_losses, valid_losses), 1):
    print(f"Epoch {epoch}: Train Loss = {t_loss:.4f}, Valid Loss = {v_loss:.4f}")

# Plotting training/validation loss
plot_result(train_losses, valid_losses)

# Load best model for prediction
print("\nLoading best model for testing...")
model.load_state_dict(torch.load("best-model.pt"))

# Create a dictionary mapping ids to indices for easier lookup
test_id_to_idx = {id_val: i for i, id_val in enumerate(test_df["id"].astype(str))}

# Make predictions on test data
ids, predictions = predict(model, test_loader, inverse_output_vocab)

# Prepare result lists
actual_values = []
before_values = []

# Match predictions with actual values using ids
for pred_id in ids:
    if pred_id in test_id_to_idx:
        idx = test_id_to_idx[pred_id]
        actual_values.append(test_df.iloc[idx]["after"])
        before_values.append(test_df.iloc[idx]["before"])
    else:
        # If id not found, use empty strings as placeholders
        actual_values.append("")
        before_values.append("")

# Calculate accuracy
correct = sum(1 for act, pred in zip(actual_values, predictions) if act == pred)
accuracy = correct / len(actual_values) * 100 if actual_values else 0
print(f"Test Accuracy: {accuracy:.2f}%")

# Save predictions for analysis
results_df = pd.DataFrame(
    {
        "id": ids,
        "before": before_values,
        "actual": actual_values,
        "predicted": predictions,
    }
)
results_df.to_csv("test_results.csv", index=False)

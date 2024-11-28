import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import logging

from scr.sbert import sbert
from scr.dataset import TextDataset
from scr.model import Model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
MODEL_SAVE_DIR = "models"
LOG_DIR = "models"
TEST_SPLIT = 0.2  # 80/20 split

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging
log_file = os.path.join(LOG_DIR, "training.log")
logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Load dataset
full_dataset = torch.load('data/dataset.pt')

# Split dataset into training and testing sets
train_size = int((1 - TEST_SPLIT) * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Initialize model
model = Model().to(device)

# Loss function and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Accuracy calculation
def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device, dtype=torch.float)
            outputs = model(texts).squeeze(1)
            predictions = (outputs > 0.5).float()  # Binary classification threshold
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total if total > 0 else 0

# Training function
def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        logger.info(f"Epoch {epoch + 1}/{epochs} started")

        # Training loop
        for texts, labels in tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            ncols=12  # Limit progress bar width to 12 characters
        ):
            texts, labels = texts.to(device), labels.to(device, dtype=torch.float)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(texts).squeeze(1)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Log average loss for the epoch
        avg_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Average loss for epoch {epoch + 1}: {avg_loss:.4f}")

        # Evaluate on the test set
        accuracy = calculate_accuracy(model, test_dataloader, device)
        logger.info(f"Test accuracy after epoch {epoch + 1}: {accuracy:.2f}%")

        # Save model
        model_path = os.path.join(MODEL_SAVE_DIR, f"model_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved: {model_path}")

    logger.info("Training complete.")

# Run training
if __name__ == "__main__":
    logger.info("Training started")
    logger.info(f"Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")
    train_model(model, train_dataloader, test_dataloader, criterion, optimizer, EPOCHS, device)
    logger.info("Training complete")

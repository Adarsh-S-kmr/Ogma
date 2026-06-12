import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AmnesiaDataset

class BaselineLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Standard LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # Linear layer to predict the next token (or in this case, the final answer)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # LSTM returns output and (hidden, cell) states
        lstm_out, _ = self.lstm(embedded)
        # We only care about the very last time step to make our prediction
        last_step_out = lstm_out[:, -1, :]
        # Predict the token
        logits = self.fc(last_step_out)
        return logits

def train_baseline():
    print("Setting up the Failing Baseline (Standard LSTM)...")
    
    # Hyperparameters
    num_samples = 10000
    batch_size = 32
    epochs = 5
    
    # 1. Load Dataset
    print(f"Generating dataset of {num_samples} samples...")
    dataset = AmnesiaDataset(num_samples=num_samples, num_pairs=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    vocab_size = dataset.vocab_size
    
    # 2. Initialize Model, Loss, and Optimizer
    model = BaselineLSTM(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    
    # 3. Training Loop
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Tracking metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.size(0)
            
        # Calculate Epoch accuracy
        accuracy = (correct_predictions / total_predictions) * 100
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
    print("\nTraining Complete.")
    print("Notice how the accuracy struggles to get high. Since the target is a digit (0-9), ")
    print("random guessing yields about 10% accuracy. The LSTM might figure out it needs to ")
    print("predict a number, but it fundamentally fails to do the associative retrieval ")
    print("required to actually look up the correct rule. This is your failing baseline!")

if __name__ == "__main__":
    train_baseline()

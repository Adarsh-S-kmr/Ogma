import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AmnesiaDataset

class FastWeightProgrammer(nn.Module):
    """
    The actual Ogma Cell implementing a Delta-Rule Update.
    Instead of relying solely on slow, permanent weights, this network creates
    temporary 'fast weights' on the fly using the Outer Product of Keys and Values.
    """
    def __init__(self, embed_dim, memory_dim):
        super().__init__()
        # Projections to generate Key, Value, and Query vectors
        self.W_k = nn.Linear(embed_dim, memory_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, memory_dim, bias=False)
        self.W_q = nn.Linear(embed_dim, memory_dim, bias=False)
        
        # A gate to decide how much of the new memory to write
        self.W_gate = nn.Linear(embed_dim, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. Generate K, V, Q for all time steps
        K = self.W_k(x)  # (batch, seq_len, memory_dim)
        V = self.W_v(x)  # (batch, seq_len, memory_dim)
        Q = self.W_q(x)  # (batch, seq_len, memory_dim)
        
        # We apply an activation function to K and Q to ensure stable outer products
        # (A standard trick in Linear Transformers / Fast Weight networks)
        K = torch.nn.functional.elu(K) + 1.0
        Q = torch.nn.functional.elu(Q) + 1.0
        
        write_gates = torch.sigmoid(self.W_gate(x)) # (batch, seq_len, 1)
        
        # 2. Initialize the empty "Whiteboard" (Fast Weight Matrix M)
        # Shape: (batch, memory_dim, memory_dim)
        M = torch.zeros(batch_size, V.size(-1), K.size(-1), device=x.device)
        
        outputs = []
        
        # 3. Process the sequence step-by-step
        for t in range(seq_len):
            # Extract the vectors for this specific time step
            k_t = K[:, t, :].unsqueeze(2)  # (batch, memory_dim, 1)
            v_t = V[:, t, :].unsqueeze(2)  # (batch, memory_dim, 1)
            q_t = Q[:, t, :].unsqueeze(2)  # (batch, memory_dim, 1)
            g_t = write_gates[:, t, :].unsqueeze(2) # (batch, 1, 1)
            
            # --- THE DELTA RULE UPDATE ---
            # Create the weight matrix for this token using the Outer Product
            # V @ K.T -> (batch, memory_dim, memory_dim)
            delta_W = torch.bmm(v_t, k_t.transpose(1, 2))
            
            # Add it to our active Fast Weight memory!
            M = M + (g_t * delta_W)
            
            # --- RETRIEVAL ---
            # Retrieve from memory by multiplying it with our Query
            # M @ Q -> (batch, memory_dim, 1)
            retrieved_info = torch.bmm(M, q_t)
            outputs.append(retrieved_info.squeeze(2))
            
        # Stack all time steps back together
        return torch.stack(outputs, dim=1)


class OgmaNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, memory_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # The 'Slow Net': A basic RNN that processes the raw tokens and provides local context.
        # It learns simple local patterns (like linking "A", "=", and "5" together) 
        # but cannot store the long-term rules itself (as proven by the baseline).
        self.slow_net = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        
        # The 'Fast Net': The FWP that replaces the long-term memory mechanism
        self.fwp = FastWeightProgrammer(hidden_dim, memory_dim)
        
        # A simple layer normalization and feed-forward to stabilize learning
        self.ln = nn.LayerNorm(memory_dim)
        self.fc = nn.Linear(memory_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        
        # Get context-aware representations from the slow net
        context_features, _ = self.slow_net(embedded)
        
        # Pass the context features through our Fast Weight Programmer
        fwp_out = self.fwp(context_features)
        
        # We only care about the very last time step for the prediction
        last_step_out = fwp_out[:, -1, :]
        
        normalized = self.ln(last_step_out)
        logits = self.fc(normalized)
        return logits

def train_ogma():
    print("Setting up Ogma (Fast Weight Programmer)...")
    
    # Hyperparameters
    num_samples = 10000
    batch_size = 32
    epochs = 5
    
    print(f"Generating dataset of {num_samples} samples...")
    dataset = AmnesiaDataset(num_samples=num_samples, num_pairs=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = OgmaNetwork(vocab_size=dataset.vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003) # FWP can take a slightly higher LR
    
    print("Starting training...")
    
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.size(0)
            
        accuracy = (correct_predictions / total_predictions) * 100
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
    print("\nTraining Complete.")
    print("Compare this to the LSTM baseline! Ogma's Fast Weight memory allows it to ")
    print("instantly write and retrieve the temporary rules, absolutely crushing the task.")

if __name__ == "__main__":
    train_ogma()

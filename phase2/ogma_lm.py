import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ShakespeareDataset

class OgmaCellLM(nn.Module):
    def __init__(self, embed_dim=64, memory_dim=64, decay=0.9):
        super().__init__()
        self.memory_dim = memory_dim
        self.decay = decay
        
        # We need to map embed_dim -> key, value, query
        self.proj_k = nn.Linear(embed_dim, memory_dim)
        self.proj_v = nn.Linear(embed_dim, memory_dim)
        self.proj_q = nn.Linear(embed_dim, memory_dim)
        
        self.layer_norm = nn.LayerNorm(memory_dim)
        self.out_proj = nn.Linear(memory_dim, embed_dim)

    def forward(self, x):
        # x is (Batch, Seq, Embed)
        B, T, E = x.shape
        device = x.device
        
        # Fast Weight memory matrix: initialized to zeros
        W = torch.zeros(B, self.memory_dim, self.memory_dim, device=device)
        
        outputs = []
        
        # Autoregressive generation over sequence
        for t in range(T):
            token_emb = x[:, t, :]  # (B, E)
            
            k = self.proj_k(token_emb)  # (B, M)
            v = self.proj_v(token_emb)  # (B, M)
            q = self.proj_q(token_emb)  # (B, M)
            
            # Normalize key and query to prevent exploding values
            k = torch.nn.functional.normalize(k, dim=-1)
            q = torch.nn.functional.normalize(q, dim=-1)
            
            # Update memory: W_t = decay * W_{t-1} + v * k^T
            # W is (B, M, M), v.unsqueeze(2) is (B, M, 1), k.unsqueeze(1) is (B, 1, M)
            W = self.decay * W + torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
            
            # Retrieve from memory: o_t = W_t * q_t
            # q.unsqueeze(2) is (B, M, 1)
            # W is (B, M, M) * (B, M, 1) -> (B, M, 1) -> squeeze to (B, M)
            read_out = torch.bmm(W, q.unsqueeze(2)).squeeze(2)
            
            read_out = self.layer_norm(read_out)
            out = self.out_proj(read_out) + token_emb # residual connection
            outputs.append(out)
            
        return torch.stack(outputs, dim=1) # (B, T, E)

class OgmaLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, memory_dim=64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Position embedding is optional since Ogma processes sequentially, 
        # but it can help. We'll add a simple one.
        self.position_embedding = nn.Embedding(256, embed_dim)
        
        self.ogma_cell = OgmaCellLM(embed_dim, memory_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        x_emb = self.token_embedding(x) + self.position_embedding(positions)
        
        ogma_out = self.ogma_cell(x_emb)
        logits = self.fc_out(ogma_out)
        return logits

def train_ogma():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    seq_len = 64
    dataset = ShakespeareDataset(seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = OgmaLM(vocab_size=dataset.vocab_size, embed_dim=64, memory_dim=64).to(device)
    
    print(f"Ogma Parameters: {sum(p.numel() for p in model.parameters())}", flush=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 1
    print("Starting training...", flush=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # Reshape for CrossEntropyLoss: (Batch * Seq_len, Vocab)
            loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}", flush=True)
                
            if batch_idx >= 50: # Just run 50 batches for testing
                break
                
        print(f"Epoch {epoch+1} Average Loss: {total_loss / 50:.4f}", flush=True)

if __name__ == "__main__":
    train_ogma()

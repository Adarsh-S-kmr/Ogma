import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ShakespeareDataset

class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Using standard PyTorch TransformerEncoder with causal mask for autoregressive generation
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len
        
    def forward(self, x):
        B, T = x.shape
        # Create causal mask (triangle mask to prevent looking ahead)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        x_emb = self.token_embedding(x) + self.position_embedding(positions)
        
        out = self.transformer(x_emb, mask=mask, is_causal=True)
        logits = self.fc_out(out)
        return logits

def train_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    seq_len = 64
    dataset = ShakespeareDataset(seq_len=seq_len)
    # Using multiple workers is tricky on Windows without __main__ guard, 
    # but we are in __main__ here, so it's safe. However, memory might be an issue.
    # We will use a smaller batch size to make testing faster.
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = TransformerBaseline(vocab_size=dataset.vocab_size, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=seq_len).to(device)
    
    print(f"Transformer Parameters: {sum(p.numel() for p in model.parameters())}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 1
    print("Starting training...")
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
            
            if batch_idx % 200 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    train_baseline()

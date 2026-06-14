import torch
from torch.utils.data import Dataset, DataLoader
import urllib.request
import os

class ShakespeareDataset(Dataset):
    def __init__(self, seq_len=64, file_path="input.txt"):
        self.seq_len = seq_len
        
        # Download dataset if it doesn't exist
        if not os.path.exists(file_path):
            print("Downloading tiny-shakespeare dataset...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, file_path)
            
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Create vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode entire text
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        print(f"Dataset loaded. Total characters: {len(self.data)}, Vocab size: {self.vocab_size}")
        
    def __len__(self):
        return len(self.data) - self.seq_len
        
    def __getitem__(self, idx):
        # We grab a chunk of sequence length
        chunk = self.data[idx:idx+self.seq_len+1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
        
    def decode(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ''.join([self.itos[i] for i in indices])

if __name__ == "__main__":
    dataset = ShakespeareDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    xb, yb = next(iter(dataloader))
    print("X shape:", xb.shape)
    print("Y shape:", yb.shape)
    print("\nSample X:")
    print(dataset.decode(xb[0]))
    print("\nSample Y (shifted by 1):")
    print(dataset.decode(yb[0]))

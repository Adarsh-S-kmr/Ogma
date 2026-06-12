try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create dummy classes to inherit from
    class Dataset: pass
    class DataLoader: pass

import random

class AmnesiaDataset(Dataset):
    """
    A synthetic dataset for the 'Amnesia' / Key-Value memory task.
    Generates sequences like "A=5, B=9, C=2, D=7, ?B" where the target is "9".
    Requires the neural network to use rapid associative retrieval instead of 
    relying on static weights learned via backpropagation.
    """
    def __init__(self, num_samples, num_pairs=4, seed=None):
        self.num_samples = num_samples
        self.num_pairs = num_pairs
        
        # Keys: A-Z
        self.keys = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        # Values: 0-9
        self.values = [str(i) for i in range(10)]
        
        # Vocabulary
        self.vocab = self.keys + self.values + ['=', ',', '?', ' ']
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        if seed is not None:
            random.seed(seed)
            
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Generate random key-value pairs
        # We sample without replacement for keys so a sequence doesn't have duplicate keys
        selected_keys = random.sample(self.keys, self.num_pairs)
        selected_values = [random.choice(self.values) for _ in range(self.num_pairs)]
        
        # Build the string representation: e.g. "A=5, B=9, C=2, D=7"
        pairs_str = [f"{k}={v}" for k, v in zip(selected_keys, selected_values)]
        sequence_str = ", ".join(pairs_str)
        
        # Pick a random key to query
        query_idx = random.randint(0, self.num_pairs - 1)
        query_key = selected_keys[query_idx]
        target_value = selected_values[query_idx]
        
        # Full input string: e.g. "A=5, B=9, C=2, D=7, ?B"
        input_str = f"{sequence_str}, ?{query_key}"
        
        # Convert to indices
        input_indices = [self.char2idx[ch] for ch in input_str]
        target_index = self.char2idx[target_value]
        
        if HAS_TORCH:
            input_tensor = torch.tensor(input_indices, dtype=torch.long)
            target_tensor = torch.tensor(target_index, dtype=torch.long)
            return input_tensor, target_tensor
        else:
            return input_indices, target_index

    def decode(self, tensor_or_list):
        """Converts an index tensor or list back to a string."""
        if HAS_TORCH and isinstance(tensor_or_list, torch.Tensor):
            if tensor_or_list.dim() == 0:
                return self.idx2char[tensor_or_list.item()]
            return "".join([self.idx2char[idx.item()] for idx in tensor_or_list])
        else:
            if isinstance(tensor_or_list, int):
                return self.idx2char[tensor_or_list]
            return "".join([self.idx2char[idx] for idx in tensor_or_list])

if __name__ == "__main__":
    # Test the dataset
    print("Testing AmnesiaDataset...")
    dataset = AmnesiaDataset(num_samples=5, num_pairs=4)
    
    if HAS_TORCH:
        # We can use DataLoader to show it works nicely in batches
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        for batch_idx, (x, y) in enumerate(dataloader):
            print(f"\n--- Batch {batch_idx+1} ---")
            for i in range(x.size(0)):
                input_text = dataset.decode(x[i])
                target_text = dataset.decode(y[i])
                print(f"Input:  {input_text}")
                print(f"Target: {target_text}")
    else:
        print("(PyTorch not found, running plain Python fallback)")
        for i in range(5):
            x, y = dataset[i]
            input_text = dataset.decode(x)
            target_text = dataset.decode(y)
            print(f"\n--- Sample {i+1} ---")
            print(f"Input:  {input_text}")
            print(f"Target: {target_text}")
            
    print(f"\nVocab size: {dataset.vocab_size}")
    if HAS_TORCH:
        print(f"Sequence length: {dataset[0][0].size(0)}")
    else:
        print(f"Sequence length: {len(dataset[0][0])}")

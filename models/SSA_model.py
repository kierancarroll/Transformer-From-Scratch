import torch.nn as nn
import torch.nn.functional as F

"""Simple Self Attention implementation"""
class SSA(nn.Module):
    def __init__(self, num_classes, vocab_size, pad_token, pool = "max", embedding_dim = 300):
        super().__init__()
        self.pad_token = pad_token

        # Define the three specific layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pool = pool
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):

        pad_mask = (x == self.pad_token) # (B,T)

        # Embedding
        embedded = self.embedding(x) # (B,T,E)

        # Simple-Self Attention
        W_prime = embedded @ embedded.transpose(1,2) # (B,T,E) x (B,E,T) = (B,T,T)
        # Add padding mask to ensure padded tokens don't contribute
        W_prime = W_prime.masked_fill(pad_mask[:, None, :], float("-inf"))
        W = F.softmax(W_prime, dim = 2) # row wise softmax over each element in batch
        attention = W @ embedded # (B,T,T) x (B,T,E) = (B,T,E)

        # Pooling
        mask = pad_mask.unsqueeze(-1) # (B, T, 1)
        if self.pool == "mean":
            attention = attention.masked_fill(mask, 0.0) # (B,T,E)
            lengths = (~pad_mask).sum(dim=1, keepdim = True) # (B,1)
            pooled = attention.mean(dim=1)/lengths # (B,E) / (B,1) = (B,E)
        elif self.pool == "max":
            attention = attention.masked_fill(mask, float("-inf")) # (B,T,E)
            pooled = attention.max(dim=1).values
        elif self.pool == "first_elem":
            pooled = attention[:,0,:]
        else:
            raise ValueError("UNKNOWN TYPE OF POOLING!")

        # Linear Layer Classifier
        out = self.linear(pooled)
        return out
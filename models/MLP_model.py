
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, num_classes, vocab_size, pool = "mean", embedding_dim = 300):
        super().__init__()

        # Define the three specific layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pool = pool
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):

        # Forward pass through the three layers, with three choices for pooling
        embedded = self.embedding(x)
        if self.pool == "mean":
            pooled = embedded.mean(dim=1)
        elif self.pool == "max":
            pooled = embedded.max(dim=1).values
        elif self.pool == "first_elem":
            pooled = embedded[:,0,:]
        else:
            raise ValueError("UNKNOWN TYPE OF POOLING!")
        out = self.linear(pooled)
        return out
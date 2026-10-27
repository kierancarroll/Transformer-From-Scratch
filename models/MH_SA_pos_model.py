import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Multi-Head Self Attention with Positional embedding model implementation"""
class MH_SA_classifier(nn.Module):
    def __init__(self, num_classes, vocab_size, num_heads, pad_token, max_seq_length, pool = "max", embedding_dim = 300):
        super().__init__()

        # Define the three specific layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear_K = nn.Linear(embedding_dim, embedding_dim)
        self.linear_Q = nn.Linear(embedding_dim, embedding_dim)
        self.linear_V = nn.Linear(embedding_dim, embedding_dim)
        self.linear_o = nn.Linear(embedding_dim, embedding_dim)
        self.pool = pool
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, num_classes)
        self.pad_token = pad_token
        self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)

    def forward(self, x, device):
        B, T = x.size() # B = batch dimension, T = time dimension
        pad_mask = (x == self.pad_token)

        # Embedding
        embedded = self.embedding(x) # (B,T,emb)

        time_dimension = torch.arange(T, device=device) # (T,)
        pos_embeddings = self.pos_embedding(time_dimension) #(T, emb)

        combined_embed = embedded+pos_embeddings # (B,T,emb) + (T, emb) -> (B,T,emb)

        K = self.linear_K(combined_embed) # (B,T,emb)
        Q = self.linear_Q(combined_embed) # (B,T,emb)
        V = self.linear_V(combined_embed) # (B,T,emb)

        hemb = self.embedding_dim // self.num_heads

        K = torch.reshape(K, (K.size(0), K.size(1), self.num_heads, hemb)) # (B,T, k, hemb)
        Q = torch.reshape(Q, (Q.size(0), Q.size(1), self.num_heads, hemb)) # (B,T, k, hemb)
        V = torch.reshape(V, (V.size(0), V.size(1), self.num_heads, hemb)) # (B,T, k, hemb)

        K = K.permute(0,2,1,3) # (B, k, T, hemb)
        Q = Q.permute(0,2,1,3) # (B, k, T, hemb)
        V = V.permute(0,2,1,3) # (B, k, T, hemb)

        Q_K = K @ Q.transpose(-1, -2) # (B, k, T, T)

        Q_K_scaled = Q_K / math.sqrt(hemb) # (B, k, T, T)

        # Add padding mask to ensure padded tokens don't contribute
        Q_K_scaled = Q_K_scaled.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        W = F.softmax(Q_K_scaled, dim=-1) # (B, k, T, T)

        Z = W @ V # (B, k, T, hemb)

        Z = Z.permute(0,2,1,3) # (B, T, k, hemb)

        Z = torch.reshape(Z, (Z.size(0), Z.size(1), self.embedding_dim)) # (B, T, emb)

        O = self.linear_o(Z) # (B, T, emb)

        # Pooling
        mask = pad_mask.unsqueeze(-1) # (B, T, 1)
        if self.pool == "mean":
          O = O.masked_fill(mask, 0.0) # (B, T, emb)
          lengths = (~pad_mask).sum(dim=1, keepdim=True) # (B, 1)
          pooled = O.sum(dim=1) / lengths # (B, emb) / (B, 1) = (B, emb)
        elif self.pool == "max":
          O = O.masked_fill(mask, float("-inf")) # (B, T, emb)
          pooled = O.max(dim=1).values  # (B, emb)
        elif self.pool == "first_elem":
          pooled = O[:,0,:]  # (B, emb)
        else:
          raise ValueError("UNKNOWN TYPE OF POOLING!")

        # Linear Layer Classifier
        logits = self.linear(pooled) # (B, num_class)
        return logits
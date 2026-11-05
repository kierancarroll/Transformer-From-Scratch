import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Multi-Head Self Attention with positional embeddings model implementation"""
class MH_SA(nn.Module):
    def __init__(self, num_heads, embedding_dim = 300):
        super().__init__()

        # Define the three specific layers
        self.linear_K = nn.Linear(embedding_dim, embedding_dim)
        self.linear_Q = nn.Linear(embedding_dim, embedding_dim)
        self.linear_V = nn.Linear(embedding_dim, embedding_dim)
        self.linear_O = nn.Linear(embedding_dim, embedding_dim)
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

    def forward(self, combined_embed, pad_mask):

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

        Q_K = Q @ K.transpose(-1, -2) # (B, k, T, T)

        Q_K_scaled = Q_K / math.sqrt(hemb) # (B, k, T, T)
        
        # Add padding mask to ensure padded tokens don't contribute
        Q_K_scaled = Q_K_scaled.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        W = F.softmax(Q_K_scaled, dim=-1) # (B, k, T, T)

        Z = W @ V # (B, k, T, hemb)

        Z = Z.permute(0,2,1,3) # (B, T, k, hemb)

        Z = torch.reshape(Z, (Z.size(0), Z.size(1), self.embedding_dim)) # (B, T, emb)

        O = self.linear_O(Z) # (B, T, emb)

        return O

class Transformer_Block(nn.Module):
  def __init__(self, num_heads, embedding_dim = 300, dropout = 0.1):
    super().__init__()
    self.selfatt = MH_SA(num_heads)
    self.ln1 = nn.LayerNorm(embedding_dim)
    self.ln2 = nn.LayerNorm(embedding_dim)
    self.ff = nn.Sequential(
        nn.Linear(embedding_dim, 4*embedding_dim),
        nn.ReLU(),
        nn.Linear(4*embedding_dim, embedding_dim)
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, combined_embed, pad_mask):
    # Sub-layer 1: LN + SelfAtt + DropOut + Resid
    combined_embed_norm = self.ln1(combined_embed)  # (B, T, emb)
    att_out = self.selfatt(combined_embed_norm, pad_mask)     # (B, T, emb)
    att_out = self.dropout(att_out)                 # (B, T, emb)
    residual_added = att_out + combined_embed       # (B, T, emb)

    # Sub-layer 2: LN + FF + DropOut + Resid
    ln_2 = self.ln2(residual_added)                 # (B, T, emb)
    ff_out = self.ff(ln_2)                          # (B, T, emb)
    ff_out = self.dropout(ff_out)                   # (B, T, emb)
    out_resid2 = ff_out + residual_added            # (B, T, emb)

    return out_resid2


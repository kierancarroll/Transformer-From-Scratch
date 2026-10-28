import torch

def batchify_rand(x, bsz, L):
  N = x.size(0)
  rand_start_indices = torch.randint(0, (N-L), (bsz,)).unsqueeze(1) # (bsz, 1)
  off_sets = torch.arange(L+1).unsqueeze(0) # (1,L+1)
  indices = rand_start_indices + off_sets # (bsz, 1) + (1,L+1) -> broadcasting
  batch = x[indices].long() # slice out each seq of batch in parallel and conver to int
  return batch

# def total_batches_per_epoch(x, bsz, L, num_batches):
#   batches = []
#   for _ in range(num_batches):
#     batches.append(batchify_rand(x, bsz, L))
#   return batches
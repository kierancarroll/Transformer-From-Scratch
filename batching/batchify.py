import torch

"""Create batches of size bsz of the sequence data and add max-length token padding"""
def batchify(data, labels, pad_token, bsz=32):
    n = len(data)
    x_batches, y_batches = [], []

    for i in range(0,n, bsz): # loop over all batches
        x_batch = [seq[-256:] for seq in data[i:i+bsz]] # slice out x_batch
        y_batch = labels[i:i+bsz] # slice out y_batch

        #max length of a sequence in batch
        max_len = max(len(seq) for seq in x_batch)
        #padd each sequence in batch with enough pad tokens to match longest sequence in length
        lists =[seq + [pad_token] * (max_len - len(seq)) for seq in x_batch]

        #convert to torch tensors and collect
        x_batch = torch.tensor(lists, dtype=torch.long)
        y_batch = torch.tensor(y_batch, dtype=torch.long)
        x_batches.append(x_batch)
        y_batches.append(y_batch)
    return x_batches, y_batches
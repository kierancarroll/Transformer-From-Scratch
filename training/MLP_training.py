import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from load_data import load_xor, load_imdb_synth, load_imdb
from batching.batchify import batchify
from models.MLP_model import LinearModel

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("Using device:", device)

epochs = 20
num_classes = 2
lr= 0.01

# (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
# (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb_synth()
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_xor()
print(f'Training samples: {len(x_train)}, Validation samples: {len(x_val)}')

pad_token = w2i['.pad']
vocab_size = len(i2w)

# visualize data
print(x_train)
print(y_train)
print(i2w)
print(w2i)
print(numcls)
print([i2w[w] for w in x_train[0]])

def training(model, optimizer, x_train_batched, y_train_batched, x_val_batched, y_val_batched, epochs):

    train_loss, val_acc = [], []

    for i in range(epochs):
      total_loss = 0
      num_correct = 0

      # train model
      model.train()

      for j, batch in enumerate(x_train_batched):
        batch = batch.to(device)
        labels = y_train_batched[j].to(device)
        optimizer.zero_grad()
        out_train = model(batch)
        loss = F.cross_entropy(out_train, labels)
        total_loss += (loss * labels.size(0)).item()
        loss.backward()
        optimizer.step()


      # evaluate model
      model.eval()
      with torch.no_grad():
        for j, batch in enumerate(x_val_batched):
          batch = batch.to(device)
          labels = y_val_batched[j].to(device)
          logits = model(batch)
          preds = logits.argmax(dim = 1)
          num_correct += (preds == labels).sum().item()


      num_data_pts_train = sum(batch.size(0) for batch in x_train_batched)
      num_data_pts_val = sum(batch.size(0) for batch in x_val_batched)
      val = num_correct/num_data_pts_val
      loss_final = total_loss/num_data_pts_train
      val_acc.append(val)
      train_loss.append(loss_final)
      print(f"Epoch {i+1} complete! | Avg train loss = {loss_final} | Val acc = {val}")


    return torch.tensor(train_loss), torch.tensor(val_acc)

x_train_batched, y_train_batched = batchify(x_train, y_train, pad_token, bsz=32)
x_val_batched, y_val_batched = batchify(x_val, y_val, pad_token, bsz=32)

model = LinearModel(num_classes, vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train
train_loss, val_acc = training(
    model,
    optimizer,
    x_train_batched,
    y_train_batched,
    x_val_batched,
    y_val_batched,
    epochs
)
epochs_range = np.arange(1, epochs+1)

# plot training loss for IMDb data
# plt.figure(figsize=(8, 5))
# plt.plot(
#         epochs_range,
#         train_loss,
#     )
# plt.title("IMDb Training Loss vs Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Training Cross Entropy Loss")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()

# plot val accuracy for IMDb data
# plt.figure(figsize=(8, 5))
# plt.plot(
#         epochs_range,
#         val_acc,
#     )
# plt.title("IMDb Validation Accuracy vs Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Validation Accuracy")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()

# perform pooling methods comparison
pooling_methods = ["mean", "max", "first_elem"]
results = {}

for pool in pooling_methods:
    print(f"\nRunning pooling = {pool}")

    # Build new model each time
    model = LinearModel(num_classes, vocab_size, pool=pool).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    train_loss, val_acc = training(
        model,
        optimizer,
        x_train_batched,
        y_train_batched,
        x_val_batched,
        y_val_batched,
        epochs
    )

    results[pool] = {
        "train_loss": train_loss.cpu(),
        "val_acc": val_acc.cpu(),
    }


plt.figure(figsize=(8, 5))
for pool in pooling_methods:
    plt.plot(
        epochs_range,
        results[pool]["train_loss"],
        label=f"pool={pool}"
    )
plt.title("XOR Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Training Cross Entropy Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
for pool in pooling_methods:
    plt.plot(
        epochs_range,
        results[pool]["val_acc"],
        label=f"pool={pool}"
    )
plt.title("XOR Validation Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
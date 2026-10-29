import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from load_data import load_xor, load_imdb_synth, load_imdb
from batching.batchify import batchify
from models.SSA_pos_model import SSA_pos_emb

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("Using device:", device)

"""Training of Simple Self Attention model with positional embeddings on XOR dataset with select pooling."""
def training(model, optimizer, x_train_batched, y_train_batched, x_val_batched, y_val_batched, epochs):
    train_loss, val_acc = [], []

    for _ in range(epochs):
        total_loss = 0
        num_correct = 0

        model.train()
        for j, batch in enumerate(x_train_batched):
            batch = batch.to(device)
            labels = y_train_batched[j].to(device)
            optimizer.zero_grad()
            out_train = model(batch, device)
            loss = F.cross_entropy(out_train, labels)
            total_loss += (loss * labels.size(0)).item()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for j, batch in enumerate(x_val_batched):
                batch = batch.to(device)
                labels = y_val_batched[j].to(device)
                logits = model(batch, device)
                preds = logits.argmax(dim=1)
                num_correct += (preds == labels).sum().item()

        num_data_pts_train = sum(batch.size(0) for batch in x_train_batched)
        num_data_pts_val = sum(batch.size(0) for batch in x_val_batched)
        val = num_correct / num_data_pts_val
        loss_final = total_loss / num_data_pts_train
        val_acc.append(val)
        train_loss.append(loss_final)

    return torch.tensor(train_loss), torch.tensor(val_acc)


def experiment_lr(x_train, y_train, x_val, y_val, epochs, num_classes, pad_token, max_seq_length, batch_size_fixed=32):
    lrs = [0.0001, 0.001, 0.01, 0.1]
    results_lr = []

    for lr in lrs:
        print(f"running lr: {lr}")
        model = SSA_pos_emb(num_classes, vocab_size, pad_token, max_seq_length, pool="first_elem").to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        x_train_b, y_train_b = batchify(x_train, y_train, pad_token, bsz=batch_size_fixed)
        x_val_b, y_val_b = batchify(x_val, y_val, pad_token, bsz=batch_size_fixed)

        train_loss, val_acc = training(model, optimizer, x_train_b, y_train_b, x_val_b, y_val_b, epochs)

        results_lr.append({
            "lr": lr,
            "batch_size": batch_size_fixed,
            "train_loss": train_loss,
            "val_acc": val_acc,
        })
        print(f"train loss: {train_loss}")
        print(f"val acc: {val_acc}")
    return results_lr


def experiment_bs(x_train, y_train, x_val, y_val, epochs, num_classes, pad_token, max_seq_length, lr_fixed=0.001):
    batch_sizes = [16, 32, 64, 128]
    results_bs = []

    for bs in batch_sizes:
        print(f"running bs: {bs}")

        model = SSA_pos_emb(num_classes, vocab_size, pad_token, max_seq_length, pool="first_elem").to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr_fixed)
        x_train_b, y_train_b = batchify(x_train, y_train, pad_token, bsz=bs)
        x_val_b, y_val_b = batchify(x_val, y_val, pad_token, bsz=bs)

        train_loss, val_acc = training(model, optimizer, x_train_b, y_train_b, x_val_b, y_val_b, epochs)
        results_bs.append({
            "lr": lr_fixed,
            "batch_size": bs,
            "train_loss": train_loss,
            "val_acc": val_acc,
        })
        print(f"train loss: {train_loss}")
        print(f"val acc: {val_acc}")

    return results_bs

def plot_results(results, epochs, param_name):
    epochs_range = torch.arange(1, epochs + 1)
    plt.figure(figsize=(8,5))

    for r in results:
        param_val = r[param_name.lower()] if param_name.lower() in r else None
        val_acc = r["val_acc"]
        plt.plot(epochs_range, val_acc, label=f"{param_name}={param_val}")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"SSA with position embeddings XOR Val Accuracy vs Epoch ({param_name})")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,5))
    for r in results:
        param_val = r[param_name.lower()] if param_name.lower() in r else None
        train_loss = r["train_loss"]
        plt.plot(epochs_range, train_loss, label=f"{param_name}={param_val}")

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"SSA with position embeddings XOR Train Loss vs Epoch ({param_name})")
    plt.grid(True)
    plt.legend()
    plt.show()


def save_results_to_csv(results, filename="SSA_pos_tuning_results_XOR.csv"):
    import csv
    fieldnames = ["lr", "batch_size", "train_loss", "val_acc"]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: (r[k].tolist() if isinstance(r[k], torch.Tensor) else r[k]) for k in fieldnames}
            writer.writerow(row)
    print(f"Saved results → {filename}")


epochs = 10
num_classes = 2
max_seq_length = 256
#(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
#(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb_synth()
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_xor()
pad_token = w2i['.pad']
vocab_size = len(i2w)

# Experiment 1: learning rates
results_lr = experiment_lr(x_train, y_train, x_val, y_val, epochs, num_classes, pad_token, max_seq_length)
plot_results(results_lr, epochs, param_name="lr")

# Experiment 2: batch sizes
results_bs = experiment_bs(x_train, y_train, x_val, y_val, epochs, num_classes,pad_token, max_seq_length)
plot_results(results_bs, epochs, param_name="batch_size")

# Save all results
save_results_to_csv(results_lr + results_bs)
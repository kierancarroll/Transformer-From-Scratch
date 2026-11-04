import os, time, csv, math
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from load_data import load_toy, load_wp
from batching.batchify_rand import batchify_rand
from models.Auto_Regressive_TransformerClassifier import Auto_Regressive_TransformerClassifier
from sampling import sample_logit

LN2 = math.log(2)

def translate_to_words(data, dictionary):
    # data: 1D tensor
    return "".join(dictionary[idx.item()] for idx in data)

def sample_to_text(indices, i2b):
    raw = bytes(i2b[i.item()] for i in indices)
    return raw.decode("utf-8", errors="replace")


def evaluate(model, val, bsz, L, device, n_batches):
    """Last-token validation over n_batches random windows."""
    model.eval()
    val_losses, num_correct, num_val = [], 0, 0
    with torch.inference_mode():
        for _ in range(n_batches):
            b = batchify_rand(val, bsz, L)
            x = b[:, :-1].to(device)
            y = b[:, 1:].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                    enabled=(device.type == "cuda")):
                logits = model(x, device)
                last_logits = logits[:, -1, :]
                last_targets = y[:, -1]
            val_losses.append(F.cross_entropy(last_logits, last_targets).item())
            num_correct += (last_logits.argmax(dim=-1) == last_targets).sum().item()
            num_val += last_targets.size(0)
    model.train()
    return (sum(val_losses) / len(val_losses))/LN2, num_correct / num_val


def generate_sample(model, sample, i2c, i2b, device, n_new=200, max_seq_length=256):
    """Autoregressive sampling with a context crop so we never exceed pos-emb range."""
    model.eval()
    s = sample.long().to(device)
    with torch.inference_mode():
        for _ in range(n_new):
            s_cond = s[-max_seq_length:]                      # crop to what pos-emb can encode
            logits = model(s_cond.unsqueeze(0), device)
            next_token = sample_logit(logits[0, -1, :])
            s = torch.cat([s, next_token.unsqueeze(0)], dim=0)
    model.train()
    return translate_to_words(s, i2c), sample_to_text(s, i2b)


def save_checkpoint(path, model, optimizer, global_step):
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),   # Adam momentum/variance — essential for clean resume
        "global_step": global_step,
    }, tmp)
    os.replace(tmp, path)                       # atomic-ish: a mid-write disconnect can't corrupt the real file


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["global_step"]


def train_autoregressive_wp(
    model, optimizer, train, val, sample, i2c, i2b, device,
    total_steps=300_000,
    steps_per_epoch=50_000,
    small_val_every=10_000, small_val_batches=200,
    big_val_batches=500,
    log_every=100,
    ckpt_every=1_000,
    max_seq_length=256,
    ckpt_path="/local/data/xze439/DL_GPT_project/WP/latest.pt",
    tb_logdir="/local/data/xze439/DL_GPT_project/runs/run2",
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    writer = SummaryWriter(log_dir=tb_logdir)

    # ---- resume if a checkpoint exists ----
    if os.path.exists(ckpt_path):
        global_step = load_checkpoint(ckpt_path, model, optimizer, device)
        print(f"Resumed from step {global_step}")
    else:
        global_step = 0

    # ---- CSV metrics (survives disconnect; easy to read back with pandas) ----
    # Two files: sparse validation metrics, and dense train metrics (every log_every).
    # kind: "small" or "big", so you can filter the two validation streams when plotting
    validation_csv_path = os.path.join(os.path.dirname(ckpt_path), "val_metrics.csv")
    metrics = {"step": [], "kind": [], "val_loss": [], "val_acc": []}
    if os.path.exists(validation_csv_path):                       # resume: keep prior rows
        prev = pd.read_csv(validation_csv_path)
        for k in metrics:
            metrics[k] = prev[k].tolist()

    train_csv_path = os.path.join(os.path.dirname(ckpt_path), "train_metrics.csv")
    train_metrics = {"step": [], "train_loss": [], "grad_norm": []}
    if os.path.exists(train_csv_path):                 # resume: keep prior rows
        prev_t = pd.read_csv(train_csv_path)
        for k in train_metrics:
            train_metrics[k] = prev_t[k].tolist()

    def flush_csv():
        pd.DataFrame(metrics).to_csv(validation_csv_path, index=False)
        pd.DataFrame(train_metrics).to_csv(train_csv_path, index=False)

    # ---- timer ---- 
    start_time = time.time() 

    model.train()
    while global_step < total_steps:
        # ---- one training step ----
        batch = batchify_rand(train, bsz, L)
        x_batch = batch[:, :-1].to(device)          # (B, T)
        y_batch = batch[:, 1:].to(device)           # (B, T)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                    enabled=(device.type == "cuda")):
            logits = model(x_batch, device)             # (B, T, C)
            B, T, C = logits.size()
            loss = F.cross_entropy(logits.reshape(B * T, C), y_batch.reshape(B * T))  # all-token
        loss.backward()

        # gradient norm (global L2)
        total_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        optimizer.step()
        global_step += 1

        # ---- lightweight scalar logging ----
        if global_step % log_every == 0:
            writer.add_scalar("train/loss_alltok", loss.item()/LN2, global_step)
            writer.add_scalar("train/grad_norm", total_norm, global_step)
            train_metrics["step"].append(global_step)
            train_metrics["train_loss"].append(loss.item()/LN2)
            train_metrics["grad_norm"].append(total_norm)

        # ---- small periodic validation + sample ----
        if global_step % small_val_every == 0:
            vloss, vacc = evaluate(model, val, bsz, L, device, small_val_batches)
            writer.add_scalar("val/loss_lasttok", vloss, global_step)
            writer.add_scalar("val/acc_lasttok", vacc, global_step)
            raw_sample, utf8_sample = generate_sample(
                model, sample, i2c, i2b, device, max_seq_length=max_seq_length)
            writer.add_text("samples/raw",  raw_sample,  global_step)   # □ for non-printable bytes
            writer.add_text("samples/utf8", utf8_sample, global_step)   # multibyte decoded, � for invalid
            metrics["step"].append(global_step)
            metrics["kind"].append("small")
            metrics["val_loss"].append(vloss)
            metrics["val_acc"].append(vacc)
            print(f"[step {global_step}] small val | loss={vloss:.4f} | acc={vacc:.4f}")
            print(raw_sample)
            print(utf8_sample)

        # ---- big validation at each 'epoch' boundary ----
        if global_step % steps_per_epoch == 0:
            vloss, vacc = evaluate(model, val, bsz, L, device, big_val_batches)
            writer.add_scalar("val_big/loss_lasttok", vloss, global_step)
            writer.add_scalar("val_big/acc_lasttok", vacc, global_step)
            metrics["step"].append(global_step)
            metrics["kind"].append("big")
            metrics["val_loss"].append(vloss)
            metrics["val_acc"].append(vacc)
            epoch = global_step // steps_per_epoch
            elpased_intermediate = time.time() - start_time
            h, rem = divmod(int(elpased_intermediate), 3600)
            m, s = divmod(rem, 60)
            print(f"=== epoch {epoch} (step {global_step}) BIG val | "
                  f"loss={vloss:.4f} | acc={vacc:.4f} | run tim = {h}h {m}m {s}s ({elpased_intermediate:.1f} s) ===")

        # ---- checkpoint (+ persist CSV alongside it) ----
        if global_step % ckpt_every == 0:
            save_checkpoint(ckpt_path, model, optimizer, global_step)
            flush_csv()

    elapsed = time.time() - start_time

    save_checkpoint(ckpt_path, model, optimizer, global_step)   # final
    flush_csv()
    writer.flush()
    writer.close()


    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print("Training complete.")
    print(f"Total training time: {h}h {m}m {s}s ({elapsed:.1f} s)")
    print(f"Validation metrics CSV: {validation_csv_path}")
    print(f"Train metrics CSV:      {train_csv_path}")


# ===========================================================================
# Driver
# ===========================================================================
if __name__ == "__main__":
    # device: works on Colab (cuda) and Mac (mps)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # hyperparams
    bsz = 16
    L = 256

    # load data
    (train, val), (b2c, b2i) = load_wp()

    i2b = {idx: byte for byte, idx in b2i.items()}   # index -> original byte

    # lookup: byte value (0-255) -> contiguous index, as a small tensor
    lut = torch.zeros(256, dtype=torch.long)
    for byte, idx in b2i.items():
        lut[byte] = idx

    # vectorized remap — no Python list, no per-element .item()
    train = lut[train.long()]     # (N,) tensor indexed by (N,) tensor → (N,) tensor
    val   = lut[val.long()]


    i2c = {b2i[b]: c for b, c in b2c.items()} # index -> character
    vocab_size = len(b2c)
    print(f"vocab size: {vocab_size}")

    print(train[:100]) #indices 0 - 204 now
    print(b2c)
    print(b2i)
    print(i2c)
    print([i2c[i.item()] for i in train[:100]])

    model = Auto_Regressive_TransformerClassifier(
        num_classes=vocab_size, vocab_size=vocab_size,
        num_heads=6, num_Tblocks=6, max_seq_length=L,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # fixed sample window for tracking generation quality over training
    S = 16
    N = val.size(0)
    start = torch.randint(0, N - S, (1,)).item()
    sample = val[start:start + S]

    train_autoregressive_wp(
        model, optimizer, train, val, sample, i2c, i2b, device,
        total_steps=1000000,
        steps_per_epoch=50_000,
        small_val_every=10_000, small_val_batches=200,
        big_val_batches=500,
        ckpt_path="/local/data/xze439/DL_GPT_project/WP/latest.pt",
        tb_logdir="/local/data/xze439/DL_GPT_project/runs/run2",
    )
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import math
import logging
import os
import time
from tqdm import tqdm

# --- Configuration ---
BATCH_SIZE = 16          # Keep the same for now
BLOCK_SIZE = 64          # INCREASED: Give the model more context
MAX_EPOCHS = 0         # Let's aim for 100 again and see where the loss plateaus
LEARNING_RATE = 3e-4     # REDUCED: A smaller learning rate for a bigger model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EMBED = 128          # INCREASED: Make the model wider
NUM_HEAD = 4             # Keep the same for now
NUM_LAYER = 6            # INCREASED: Make the model deeper
DROPOUT = 0.1
DATA_FILE = 'domain_data.txt'
CHECKPOINT_DIR = 'checkpoints_v2' # Use a new directory to avoid confusion
LOG_FILE = 'training_v2.log'

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# --- Data Loading and Tokenizer ---
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.block_size = block_size
        self.data = torch.tensor(encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# --- Model Architecture ---
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(NUM_EMBED, head_size, bias=False)
        self.query = nn.Linear(NUM_EMBED, head_size, bias=False)
        self.value = nn.Linear(NUM_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(NUM_EMBED, NUM_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SmallLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, NUM_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, NUM_EMBED)
        self.blocks = nn.Sequential(*[Block(NUM_EMBED, n_head=NUM_HEAD) for _ in range(NUM_LAYER)])
        self.ln_f = nn.LayerNorm(NUM_EMBED)
        self.lm_head = nn.Linear(NUM_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Checkpointing Functions ---
def save_checkpoint(model, optimizer, epoch, loss):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        logging.warning(f"Checkpoint file not found: {checkpoint_path}. Starting from scratch.")
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logging.info(f"Resumed training from checkpoint: {checkpoint_path} at epoch {epoch+1}")
    return epoch + 1, loss


# --- Main Training Loop ---
if __name__ == '__main__':
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Vocabulary size: {vocab_size}")

    # Data Loader
    full_dataset = TextDataset(text, BLOCK_SIZE)
    # Split into train and validation sets
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model and Optimizer
    model = SmallLanguageModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Attempt to resume from the latest checkpoint
    start_epoch = 0
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')], reverse=True)
        if checkpoints:
            latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[0])
            start_epoch, _ = load_checkpoint(model, optimizer, latest_checkpoint)

    # Training
    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
        for xb, yb in progress_bar:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} | Training Loss: {avg_epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits, loss = model(xb, yb)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1} | Validation Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch + 1, avg_epoch_loss)

    logging.info("Training finished.")

    # --- Generation ---
    logging.info("Generating text from the trained model...")
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    logging.info(f"Generated Text:\n{generated_text}")
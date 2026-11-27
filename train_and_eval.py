import re
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

# ==========================================
# 1. Global Parameters & Setup
# ==========================================
CSV_PATH = "clinical_notes.csv"  # change to whatever dataset is needed
TEXT_COL = "note"           # can also switch to "gpt4_summary"
LABEL_COL = "glaucoma"
RACE_COL = "race"

MAX_VOCAB = 10000
MAX_LEN = 256
BATCH_SIZE = 32
EMBED_DIM = 128
HIDDEN_DIM = 128
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())

class ClinicalNotesDataset(Dataset):
    def __init__(self, texts, labels, races, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.races = races
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx])
        ids = [self.vocab.get(t, 1) for t in tokens][:self.max_len]  # 1 = <UNK>
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))  # 0 = <PAD>
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            self.races[idx]
        )

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    
    # Map 'yes'/'no' to 1/0
    df["target"] = df[LABEL_COL].str.strip().str.lower().map({"yes": 1, "no": 0})
    df[RACE_COL] = df[RACE_COL].fillna("unknown").str.strip().str.lower()

    # Check if 'use' column has test/val splits; otherwise do an 80/20 split
    splits = df["use"].str.strip().str.lower().unique() if "use" in df.columns else []
    if "test" in splits or "validation" in splits:
        train_df = df[df["use"].str.strip().str.lower() == "training"]
        test_df = df[df["use"].str.strip().str.lower() != "training"]
    else:
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["target"]
        )

    # Build vocabulary on training set only
    counter = Counter()
    for text in train_df[TEXT_COL]:
        counter.update(tokenize(text))
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(MAX_VOCAB - 2):
        vocab[word] = len(vocab)

    train_ds = ClinicalNotesDataset(
        train_df[TEXT_COL].tolist(), train_df["target"].tolist(), train_df[RACE_COL].tolist(), vocab, MAX_LEN
    )
    test_ds = ClinicalNotesDataset(
        test_df[TEXT_COL].tolist(), test_df["target"].tolist(), test_df[RACE_COL].tolist(), vocab, MAX_LEN
    )
    return train_ds, test_ds, len(vocab)

# ==========================================
# 3. Benchmark Models (1D CNN & GRU)
# ==========================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters=100, kernel_sizes=(3, 4, 5)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        # x: [batch, seq_len] -> embedded: [batch, embed_dim, seq_len]
        emb = self.embedding(x).permute(0, 2, 1)
        pooled = [torch.relu(conv(emb)).max(dim=2)[0] for conv in self.convs]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat).squeeze(1)

class TextGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        _, hidden = self.gru(emb)
        # Concatenate final forward and backward hidden states
        cat = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        return self.fc(cat).squeeze(1)

# ==========================================
# 4. Training & Evaluation Pipeline
# ==========================================
def train_model(model, train_loader):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader, model_name):
    model.eval()
    all_probs, all_labels, all_races = [], [], []

    with torch.no_grad():
        for x, y, races in test_loader:
            x = x.to(DEVICE)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
            all_races.extend(races)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_races = np.array(all_races)
    preds = (all_probs >= 0.5).astype(int)

    # Overall Metrics
    overall_auc = roc_auc_score(all_labels, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n================ {model_name} Results ================")
    print(f"Overall AUC : {overall_auc:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")

    # Subgroup AUCs
    for group in ["asian", "black", "white"]:
        mask = (all_races == group)
        if mask.sum() > 0 and len(np.unique(all_labels[mask])) == 2:
            group_auc = roc_auc_score(all_labels[mask], all_probs[mask])
            print(f"AUC ({group.capitalize():<5}): {group_auc:.4f} (n={mask.sum()})")
        else:
            print(f"AUC ({group.capitalize():<5}): N/A (insufficient class diversity, n={mask.sum()})")
    print("=====================================================\n")

if __name__ == "__main__":
    train_ds, test_ds, vocab_size = prepare_data(CSV_PATH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("Training 1D CNN...")
    cnn_model = TextCNN(vocab_size, EMBED_DIM)
    train_model(cnn_model, train_loader)
    evaluate_model(cnn_model, test_loader, "1D CNN")

    print("Training Bidirectional GRU...")
    gru_model = TextGRU(vocab_size, EMBED_DIM, HIDDEN_DIM)
    train_model(gru_model, train_loader)
    evaluate_model(gru_model, test_loader, "GRU")

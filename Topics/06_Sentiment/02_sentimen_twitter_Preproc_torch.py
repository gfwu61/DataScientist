# -*- coding: utf-8 -*-
"""
PyTorch Sentiment LSTM - NO EXTERNAL DEPS! (torchtext-free)
100% self-contained - läuft überall! + EPOCHEN-DAUER MESSUNG
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import time  # ← NEU: Für Epoche-Dauer-Messung

print('🚀 PURE PYTORCH SENTIMENT LSTM - NO torchtext needed!')
print('─' * 80)

# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'✅ Device: {device}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')

# Hyperparameter
sequence_length = 25
num_words = 10000
batch_size = 500
epochs = 10
embedding_dim = 3
hidden_dim1 = 64
hidden_dim2 = 32

print('\n' + '─'*50 + ' 1. DATA LOADING ' + '─'*50)

# Deine CSV-Datei laden
data = pd.read_csv('sentiment_Analyse.csv', delimiter=',', encoding='utf-8')
print(f'📊 Total datapoints: {len(data)}')

y = data.iloc[:, 0].values.astype('float32') / 4  # Labels 0/1
X = data.iloc[:, 5].values  # Tweets

print('First 3 raw examples:')
for i in range(3):
    print(f'  {X[i][:60]}...')

print('\n' + '─'*50 + ' 2. TEXT CLEANING ' + '─'*50)

def clean_text(text):
    """Identisch zum Original - ohne Regex-Abhängigkeiten"""
    # @-Usernames entfernen
    words = text.split()
    words = [w for w in words if not w.startswith('@')]
    text = ' '.join(words)
    
    # XML-Ersetzungen (hartcodiert)
    replacements = {
        '&ampPOOOOOLL': ' ', '&quotPOOOOOLL': ' ', 
        '&ltPOOOOOLL': ' ', '&gtPOOOOOLL': ' ', 'POOOOOLL': ' '
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.strip().lower()

X_clean = [clean_text(text) for text in X]
print('✅ First 5 cleaned:')
for text in X_clean[:5]:
    print(f'  {repr(text[:60])}...')

# Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(
    X_clean, y, test_size=0.2, random_state=0
)

print(f'✅ Train: {len(X_train):,} | Val: {len(X_val):,}')

print('\n' + '─'*50 + ' 3. TOKENIZER (100% CUSTOM) ' + '─'*50)

def simple_tokenizer(text):
    """Custom tokenizer - splits on whitespace & punctuation"""
    # Einfache Tokenisierung: whitespace + punctuation
    text = re.sub(r'[^\w\s]', ' ', text)  # Punctuation -> space
    return text.split()

# Vocabulary bauen
all_tokens = []
for text in X_train:
    all_tokens.extend(simple_tokenizer(text))

word_counts = Counter(all_tokens)
vocab = ['<pad>', '<unk>'] + [word for word, _ in word_counts.most_common(num_words-2)]
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

vocab_size = len(vocab)
print(f'✅ Vocab size: {vocab_size:,}')
print(f'   Top 10: {vocab[2:12]}')  # Skip pad/unk

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_len):
        self.texts = texts
        self.labels = torch.FloatTensor(labels)
        self.word_to_idx = word_to_idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = simple_tokenizer(self.texts[idx])
        indices = [self.word_to_idx.get(token, 1) for token in tokens]  # 1 = <unk>
        indices = indices[:self.max_len] + [0] * (self.max_len - len(indices))  # 0 = <pad>
        return torch.LongTensor(indices), self.labels[idx]

# DataLoader
train_ds = SentimentDataset(X_train, y_train, word_to_idx, sequence_length)
val_ds = SentimentDataset(X_val, y_val, word_to_idx, sequence_length)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

print('✅ DataLoaders ready! Sample batch:')
sample_text, sample_label = next(iter(train_loader))
print(f'   Batch shapes: texts={sample_text.shape}, labels={sample_label.shape}')

print('\n' + '─'*50 + ' 4. LSTM MODEL ' + '─'*50)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden1, hidden2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embed_dim, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc = nn.Linear(hidden2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        out1, _ = self.lstm1(embedded)
        out2, _ = self.lstm2(out1)
        pooled = torch.mean(out2, dim=1)
        return self.sigmoid(self.fc(pooled))
# vocab_size=10000, embedding_dim = 3, hidden_dim1 = 64, hidden_dim2 = 32
model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim1, hidden_dim2).to(device)
print(model)
print(f'💾 Params: {sum(p.numel() for p in model.parameters()):,}')

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print('\n' + '─'*50 + ' 5. TRAINING mit EPOCHEN-DAUER ' + '─'*50)

best_val_acc = 0
train_losses, val_losses = [], []
train_accs, val_accs = [], []
epoch_durations = []  # ← NEU: Dauer pro Epoche
total_start_time = time.time()  # ← NEU: Gesamtstart

for epoch in range(epochs):
    epoch_start = time.time()  # ← NEU: Epoche-Start
    
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        preds = model(texts).squeeze()
        loss = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        train_loss += loss.item()
        train_correct += ((preds > 0.5) == (labels > 0.5)).sum().item()
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            preds = model(texts).squeeze()
            loss = criterion(preds, labels)
            
            val_loss += loss.item()
            val_correct += ((preds > 0.5) == (labels > 0.5)).sum().item()
    
    # Metrics
    train_acc = train_correct / len(train_ds)
    val_acc = val_correct / len(val_ds)
    
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    
    # ← NEU: Epoche-Dauer berechnen
    epoch_duration = time.time() - epoch_start
    epoch_durations.append(epoch_duration)
    total_duration = time.time() - total_start_time
    avg_epoch_time = np.mean(epoch_durations)
    
    print(f' Epoch {epoch+1:2d}: Train={train_acc:.1%}({train_losses[-1]:.4f}) | '
          f'Val={val_acc:.1%}({val_losses[-1]:.4f}) | Best={best_val_acc:.1%} | '
          f'⏱️ {epoch_duration:.1f}s (Ø{avg_epoch_time:.1f}s)')

print(f'\n🏆 FINAL RESULTS:')
print(f'   Training Acc:  {train_accs[-1]:.1%}')
print(f'   Validation Acc: {val_accs[-1]:.1%}')
print(f'   Best Val Acc:   {best_val_acc:.1%}')
print(f'   ⏱️  Total Time:  {total_duration:.1f}s')
print(f'   ⏱️  Ø Epoch:     {np.mean(epoch_durations):.1f}s')

print('\n' + '─'*50 + ' 6. EMBEDDINGS VISUALIZE ' + '─'*50)

with torch.no_grad():
    embeddings = model.embedding.weight.cpu().numpy()

print('Embeddings shape:', embeddings.shape)
print('Sample embeddings:')
for i in range(10):
    word = idx_to_word.get(i, f'#{i}')
    print(f'  {word:8s}: {embeddings[i][:3]}')

# 3D Plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
n_show = min(30, vocab_size)

ax.scatter(embeddings[:n_show, 0], embeddings[:n_show, 1], embeddings[:n_show, 2], 
           c='red', s=60, alpha=0.8)
ax.view_init(elev=10, azim=45)

for i in range(n_show):
    if i < len(idx_to_word):
        word = idx_to_word[i]
        ax.text(embeddings[i, 0], embeddings[i, 1], embeddings[i, 2], 
                word[:8], fontsize=9)

ax.set_title('Word Embeddings (3D)')
ax.set_xlabel('Dim 0'); ax.set_ylabel('Dim 1'); ax.set_zlabel('Dim 2')
plt.show()

print('\n' + '─'*50 + ' 7. INFERENCE FUNCTION ' + '─'*50)

def predict(texts):
    """Ready-to-use Prediction!"""
    model.eval()
    results = []
    with torch.no_grad():
        for text in texts:
            tokens = simple_tokenizer(clean_text(text))
            indices = [word_to_idx.get(t, 1) for t in tokens][:sequence_length]
            indices += [0] * (sequence_length - len(indices))
            inp = torch.LongTensor([indices]).to(device)
            pred = model(inp).cpu().item()
            results.append(pred)
    return results

# Test
test_cases = [
    'bad', 'sad', 'good', 'happy', 'great',
    'I love this!', 'I hate it', 'not satisfied', 'I love Trump'
]

print('🧪 Predictions:')
for text, score in zip(test_cases, predict(test_cases)):
    label = 'POS' if score > 0.5 else 'NEG'
    print(f'  {text:25s} → {score:.3f} ({label})')

print('\n' + '─'*50 + ' 8. TRAINING CURVES + EPOCH TIMES ' + '─'*50)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Loss & Accuracy
ax1.plot(train_losses, 'b-', lw=2, label='Train')
ax1.plot(val_losses, 'r-', lw=2, label='Val')
ax1.set_title('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, 'b-', lw=2, label='Train')
ax2.plot(val_accs, 'r-', lw=2, label='Val')
ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)

# ← NEU: Epoche-Dauern Plot
ax3.bar(range(1, epochs+1), epoch_durations, color='green', alpha=0.7)
ax3.set_title('Epoch Duration'); ax3.set_xlabel('Epoch'); ax3.set_ylabel('Time (s)')
ax3.grid(True, alpha=0.3)

# ← NEU: Kumulative Zeit
cum_times = np.cumsum(epoch_durations)
ax4.plot(range(1, epochs+1), cum_times, 'purple', lw=3, marker='o')
ax4.set_title('Cumulative Training Time'); ax4.set_xlabel('Epoch'); ax4.set_ylabel('Total Time (s)')
ax4.grid(True, alpha=0.3)

plt.suptitle('Training History + Timing Analysis')
plt.tight_layout()
plt.show()

print('\n🎉 DONE! Model ist ready für Production!')
print(f'   Save with: torch.save(model.state_dict(), "sentiment_lstm.pth")')
print(f'   ⏱️  Detailed timing saved in epoch_durations list')
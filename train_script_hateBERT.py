import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import os
import random
import csv
import requests
from collections import Counter

# Config & Reproducibility

SEED = 42
MODEL_NAME = 'GroNLP/hateBERT'
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
LAMBDA_RAT = 1 
SAVE_DIR = "/content/gdrive/MyDrive/models/CAP_model(lamda 3)"
LOG_PATH = os.path.join(SAVE_DIR, "training_log.csv")

LABEL2ID = {"normal": 0, "offensive": 1, "hatespeech": 2}
NUM_LABELS = len(LABEL2ID)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Download & Split HateXplain Dataset

BASE_URL = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/"

print("Downloading HateXplain dataset...")
dataset = requests.get(BASE_URL + "dataset.json").json()

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
MIN_ANNOTATOR_AGREEMENT = 2

all_post_ids = list(dataset.keys())
random.seed(SEED)
random.shuffle(all_post_ids)

total = len(all_post_ids)
train_end = int(total * TRAIN_RATIO)
val_end   = train_end + int(total * VAL_RATIO)

train_ids = all_post_ids[:train_end]
val_ids   = all_post_ids[train_end:val_end]
test_ids  = all_post_ids[val_end:]

def load_ids_to_df(id_list):
    rows = []
    for tweet_id in id_list:
        info = dataset[tweet_id]
        post_tokens = info["post_tokens"]

        label_counts = Counter(a["label"] for a in info["annotators"])
        if not label_counts:
            continue

        final_label = label_counts.most_common(1)[0][0]
        if final_label not in LABEL2ID:
            continue

        if MIN_ANNOTATOR_AGREEMENT > 1:
            majority_count = sum(1 for a in info["annotators"] if a["label"] == final_label)
            if majority_count < MIN_ANNOTATOR_AGREEMENT:
                continue

        consensus_rationale = [0] * len(post_tokens)

        if "rationales" in info and info["rationales"]:
            try:
                rationale_matrix = np.array(info["rationales"])
                n_annotators_with_rationales = np.sum(np.any(rationale_matrix, axis=1))
                if n_annotators_with_rationales > 0:
                    token_sums = np.sum(rationale_matrix, axis=0)
                    threshold = 0.5 * n_annotators_with_rationales
                    consensus_rationale = [1 if s >= threshold else 0 for s in token_sums]
            except ValueError:
                pass

        rows.append({
            "post_id": tweet_id,
            "majority_label": final_label,
            "post_tokens": post_tokens,
            "consensus_rationale": consensus_rationale
        })

    return pd.DataFrame(rows)

print("Processing splits into DataFrames...")
df_train = load_ids_to_df(train_ids)
df_val   = load_ids_to_df(val_ids)
df_test  = load_ids_to_df(test_ids)

def verify_alignment(df):
    bad_idx = []
    for i, row in df.iterrows():
        if len(row['post_tokens']) != len(row['consensus_rationale']):
            bad_idx.append(i)
    if bad_idx:
        df = df.drop(index=bad_idx).reset_index(drop=True)
    return df

def drop_empty_rows(df):
    df = df[df['post_tokens'].apply(len) > 0].reset_index(drop=True)
    return df

df_train = drop_empty_rows(verify_alignment(df_train))
df_val   = drop_empty_rows(verify_alignment(df_val))
df_test  = drop_empty_rows(verify_alignment(df_test))

train_words   = df_train['post_tokens'].tolist()
train_rats    = df_train['consensus_rationale'].tolist()
train_labels  = [LABEL2ID[lbl] for lbl in df_train['majority_label'].tolist()]

val_words     = df_val['post_tokens'].tolist()
val_rats      = df_val['consensus_rationale'].tolist()
val_labels    = [LABEL2ID[lbl] for lbl in df_val['majority_label'].tolist()]

test_words    = df_test['post_tokens'].tolist()
test_rats     = df_test['consensus_rationale'].tolist()
test_labels   = [LABEL2ID[lbl] for lbl in df_test['majority_label'].tolist()]

print(f"Final Count -> Train: {len(train_labels)} | Val: {len(val_labels)} | Test: {len(test_labels)}")

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Token-Aligned Dataset

class TokenAlignedDataset(Dataset):
    def __init__(self, words_list, rationales_list, labels, tokenizer, max_length=MAX_LENGTH):
        self.words_list = words_list
        self.rationales_list = rationales_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        words = self.words_list[idx]
        word_rationales = self.rationales_list[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        word_ids = encoding.word_ids(batch_index=0)

        token_valid_mask = torch.zeros(self.max_length, dtype=torch.float)
        token_rationale_mask = torch.zeros(self.max_length, dtype=torch.float)

        for seq_idx, w_id in enumerate(word_ids):
            # Ignore [CLS], [SEP], and [PAD] (where w_id is None)
            if w_id is not None:
                token_valid_mask[seq_idx] = 1.0
                if w_id < len(word_rationales):
                    token_rationale_mask[seq_idx] = float(word_rationales[w_id])

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'token_valid_mask': token_valid_mask,
            'token_rationale_mask': token_rationale_mask
        }

'''
---------------------------------------------------------
CAP Architecture: Single Shared Linear Head
---------------------------------------------------------
One linear layer maps each token's hidden state to per-token,
per-class logits. Classification and rationale supervision are
two READOUTS OF THE SAME LOGITS, not two separately-parameterized
heads:

  token_logits[i, c]  = head(h_i)[c]                    (shared)
  sequence_logits[c]  = mean_i( token_logits[i, c] )     (softmax -> classification)
  rationale_logit[i]  = token_logits[i, y]               (sigmoid -> token rationale, class y)

Because sequence_logits is a plain mean of token_logits, each
token's rationale score (sigmoid of its own logit at the target
class) is a monotonic transform of that token's exact additive
contribution to the classification decision -- faithfulness by
construction, not by auxiliary loss alignment.
---------------------------------------------------------
'''

class CAPModel(nn.Module):
    def __init__(self, model_name, num_labels=3, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.num_labels = num_labels

        self.dropout = nn.Dropout(dropout)
        # SINGLE SHARED HEAD -- the only linear layer producing
        # class-relevant logits anywhere in the model.
        self.head = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_valid_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        subword_states = self.dropout(outputs.last_hidden_state)  # [B, L, H]

        # Shared head applied per-token -> per-token, per-class logits
        token_logits = self.head(subword_states)  # [B, L, C]

        # Zero out special/padding token logits before pooling
        valid_mask = token_valid_mask.unsqueeze(-1)  # [B, L, 1]
        masked_token_logits = token_logits * valid_mask

        # --- Classification readout: mean-pool over valid tokens ---
        valid_counts = token_valid_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)  # [B, 1]
        sequence_logits = masked_token_logits.sum(dim=1) / valid_counts  # [B, C]

        # Return raw per-token, per-class logits; the rationale
        # readout (sigmoid at a specific class index) is computed
        # by the caller, since which class to index depends on
        # gold label (training) vs. predicted label (inference).
        return sequence_logits, token_logits
'''
---------------------------------------------------------
Rationale Loss (reads out the SAME logits used for
   classification, indexed at the gold class)
---------------------------------------------------------
'''

def token_rationale_loss(token_logits, labels, token_rationale_mask, token_valid_mask):
    """
    token_logits: [B, L, C] -- shared-head logits (same tensor used
                  to compute sequence_logits via mean-pooling)
    labels:       [B]       -- gold class index per example
    """
    B, L, C = token_logits.shape

    rationale_sum = token_rationale_mask.sum(dim=1)
    has_rationale = (rationale_sum > 0)

    if has_rationale.sum() == 0:
        return torch.tensor(0.0, device=token_logits.device)

    # Gather each token's logit at the GOLD class -> [B, L]
    gold_idx = labels.view(B, 1, 1).expand(B, L, 1)
    gold_token_logits = token_logits.gather(dim=2, index=gold_idx).squeeze(-1)  # [B, L]

    pred_logits = gold_token_logits[has_rationale]
    targets = token_rationale_mask[has_rationale]
    valid_tokens = token_valid_mask[has_rationale]

    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    loss = bce_loss_fn(pred_logits, targets)
    masked_loss = loss * valid_tokens

    denom = valid_tokens.sum()
    if denom == 0:
        return torch.tensor(0.0, device=token_logits.device)

    return masked_loss.sum() / denom
#Evaluate taring just to pick the best model

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    token_f1_scores, iou_scores = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_valid_mask = batch['token_valid_mask'].to(device)
            gold_rationale = batch['token_rationale_mask'].to(device)

            sequence_logits, token_logits = model(input_ids, attention_mask, token_valid_mask)

            preds = torch.argmax(sequence_logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            # Read out rationale at the PREDICTED class (no oracle leakage)
            B, L, C = token_logits.shape
            pred_idx = preds.view(B, 1, 1).expand(B, L, 1)
            pred_token_logits = token_logits.gather(dim=2, index=pred_idx).squeeze(-1)  # [B, L]
            token_probs = torch.sigmoid(pred_token_logits)
            pred_rationale_mask = (token_probs > 0.5).float()

            for i in range(input_ids.size(0)):
                gold = gold_rationale[i]
                tm = token_valid_mask[i]

                if gold.sum().item() == 0:
                    continue

                valid = tm.bool()
                p_mask = pred_rationale_mask[i][valid].bool()
                g_mask = gold[valid].bool()

                tp = (p_mask & g_mask).sum().item()
                fp = (p_mask & ~g_mask).sum().item()
                fn = (~p_mask & g_mask).sum().item()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                token_f1_scores.append(f1)

                intersection = (p_mask & g_mask).sum().item()
                union = (p_mask | g_mask).sum().item()
                iou = (intersection / union) if union > 0 else (1.0 if g_mask.sum().item() == 0 else 0.0)
                iou_scores.append(iou)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    mean_token_f1 = float(np.mean(token_f1_scores)) if token_f1_scores else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    return acc, macro_f1, mean_token_f1, mean_iou, all_preds, all_labels


# Setup & Training Loop

print(f"Loading {MODEL_NAME} tokenizer and initializing CAP model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = CAPModel(MODEL_NAME, num_labels=NUM_LABELS).to(device)

train_dataset = TokenAlignedDataset(train_words, train_rats, train_labels, tokenizer)
val_dataset   = TokenAlignedDataset(val_words, val_rats, val_labels, tokenizer)
test_dataset  = TokenAlignedDataset(test_words, test_rats, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

with open(LOG_PATH, 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'train_loss', 'cls_loss', 'rat_loss', 'val_acc', 'val_macro_f1', 'val_token_f1', 'val_iou'])

best_val_f1 = -1.0
best_model_path = os.path.join(SAVE_DIR, "best_cap_model.pt")

print(f"Starting CAP training on {device}...")

for epoch in range(EPOCHS):
    model.train()
    total_loss, total_cls_loss, total_rat_loss = 0.0, 0.0, 0.0
    n_valid_batches = 0

    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        token_valid_mask = batch['token_valid_mask'].to(device)
        token_rationale_mask = batch['token_rationale_mask'].to(device)

        optimizer.zero_grad()

        sequence_logits, token_logits = model(input_ids, attention_mask, token_valid_mask)

        cls_loss = cls_loss_fn(sequence_logits, labels)
        rat_loss = token_rationale_loss(token_logits, labels, token_rationale_mask, token_valid_mask)
        loss = cls_loss + (LAMBDA_RAT * rat_loss)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_rat_loss += rat_loss.item()
        n_valid_batches += 1

        loop.set_description(f'Epoch {epoch+1}')
        loop.set_postfix(Loss=loss.item(), Cls=cls_loss.item(), RatBCE=rat_loss.item())

    denom = max(n_valid_batches, 1)
    avg_loss = total_loss / denom
    avg_cls = total_cls_loss / denom
    avg_rat = total_rat_loss / denom

    val_acc, val_macro_f1, val_token_f1, val_iou, _, _ = evaluate(model, val_loader)

    print(f"\nEpoch {epoch+1} | Loss: {avg_loss:.4f} | Cls: {avg_cls:.4f} | RatBCE: {avg_rat:.4f}")
    print(f"Validation -> Acc: {val_acc:.4f} | Macro-F1: {val_macro_f1:.4f} | Token-F1: {val_token_f1:.4f} | IOU: {val_iou:.4f}\n")

    with open(LOG_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([epoch+1, avg_loss, avg_cls, avg_rat, val_acc, val_macro_f1, val_token_f1, val_iou])

    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved based on Macro-F1: {val_macro_f1:.4f}")

print("Training complete!")

# Evaluation

model.load_state_dict(torch.load(best_model_path))
test_acc, test_macro_f1, test_token_f1, test_iou, test_preds, test_labels_out = evaluate(model, test_loader)

print(f"\n" + "="*60)
print(f"FINAL TEST SET RESULTS (CAP: Single Shared Linear Head)")
print(f"="*60)
print(f"Sentence Accuracy      : {test_acc:.4f}")
print(f"Sentence Macro-F1      : {test_macro_f1:.4f}")
print(f"Subword-Token F1       : {test_token_f1:.4f}")
print(f"Subword-Token IOU      : {test_iou:.4f}")
print(f"="*60)
print(classification_report(test_labels_out, test_preds, target_names=list(LABEL2ID.keys()), digits=4))

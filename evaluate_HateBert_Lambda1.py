#!/usr/bin/env python3
"""
Evaluation script for the CAP model (Single Shared Linear Head).
Loads the trained CAP model from Hugging Face Hub.
"""

import os
import sys
import json
import argparse
import warnings
import random
from itertools import groupby
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel, PretrainedConfig
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import requests

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# Custom model classes (as in modeling_cap.py)
# -------------------------------------------------------------------
class CAPConfig(PretrainedConfig):
    model_type = "cap"

    def __init__(self, base_model_name="roberta-base", num_labels=3, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.dropout = dropout


class CAPModel(PreTrainedModel):
    config_class = CAPConfig
    base_model_prefix = "backbone"

    def __init__(self, config):
        super().__init__(config)
        backbone_config = AutoConfig.from_pretrained(config.base_model_name)
        self.backbone = AutoModel.from_config(backbone_config)
        hidden_size = self.backbone.config.hidden_size
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids, attention_mask, token_valid_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        subword_states = self.dropout(outputs.last_hidden_state)
        token_logits = self.head(subword_states)
        valid_mask = token_valid_mask.unsqueeze(-1)
        masked_token_logits = token_logits * valid_mask
        valid_counts = token_valid_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        sequence_logits = masked_token_logits.sum(dim=1) / valid_counts
        return sequence_logits, token_logits


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
class Config:
    SEED = 42
    MODEL_NAME = 'GroNLP/hateBERT'          # backbone used for tokenizer
    HF_MODEL_ID = "anonymous-CAP/Hate_bert_lambda1"   # your Hugging Face model
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    NUM_LABELS = 3
    LABEL_MAPPING = {'normal': 0, 'offensive': 1, 'hatespeech': 2}
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split parameters (same as training)
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    MIN_ANNOTATOR_AGREEMENT = 2

config = Config()

# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(config.SEED)

# -------------------------------------------------------------------
# Dataset (subword-token aligned) – unchanged
# -------------------------------------------------------------------
class TokenAlignedDataset(Dataset):
    def __init__(self, words_list, rationales_list, labels, tokenizer, max_length=128):
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
            if w_id is not None:                # ignore [CLS], [SEP], [PAD]
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

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    token_valid_mask = torch.stack([item['token_valid_mask'] for item in batch])
    token_rationale_mask = torch.stack([item['token_rationale_mask'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'token_valid_mask': token_valid_mask,
        'token_rationale_mask': token_rationale_mask
    }

# -------------------------------------------------------------------
# Data loading & split (same as training script) – unchanged
# -------------------------------------------------------------------
def load_and_split_data():
    BASE_URL = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/"
    dataset = requests.get(BASE_URL + "dataset.json").json()

    all_post_ids = list(dataset.keys())
    random.seed(config.SEED)
    random.shuffle(all_post_ids)

    total = len(all_post_ids)
    train_end = int(total * config.TRAIN_RATIO)
    val_end = train_end + int(total * config.VAL_RATIO)
    test_ids = all_post_ids[val_end:]

    def load_ids(id_list):
        rows = []
        for tweet_id in id_list:
            info = dataset[tweet_id]
            post_tokens = info["post_tokens"]

            label_counts = Counter(a["label"] for a in info["annotators"])
            if not label_counts:
                continue
            final_label = label_counts.most_common(1)[0][0]
            if final_label not in config.LABEL_MAPPING:
                continue
            if config.MIN_ANNOTATOR_AGREEMENT > 1:
                majority_count = sum(1 for a in info["annotators"] if a["label"] == final_label)
                if majority_count < config.MIN_ANNOTATOR_AGREEMENT:
                    continue

            # Consensus rationale (≥50% of annotators with any rationale)
            consensus_rationale = [0] * len(post_tokens)
            if "rationales" in info and info["rationales"]:
                try:
                    rationale_matrix = np.array(info["rationales"])
                    n_annot = np.sum(np.any(rationale_matrix, axis=1))
                    if n_annot > 0:
                        token_sums = np.sum(rationale_matrix, axis=0)
                        threshold = 0.5 * n_annot
                        consensus_rationale = [1 if s >= threshold else 0 for s in token_sums]
                except ValueError:
                    pass

            # Target communities (for bias metrics)
            targets_all = []
            for ann in info['annotators']:
                if isinstance(ann, dict) and 'target' in ann:
                    t = ann['target']
                    if isinstance(t, list):
                        targets_all.extend(t)
                    elif isinstance(t, str) and t != 'None':
                        targets_all.append(t)
            community_counts = Counter(targets_all)
            final_comms = [c for c, cnt in community_counts.items() if cnt >= 2 and c not in ['None', 'Other']]
            final_target_category = final_comms if final_comms else None

            rows.append({
                "post_id": tweet_id,
                "majority_label": final_label,
                "post_tokens": post_tokens,
                "consensus_rationale": consensus_rationale,
                "final_target_category": final_target_category
            })
        return pd.DataFrame(rows)

    df_test = load_ids(test_ids)

    def verify_alignment(df):
        bad = [i for i, row in df.iterrows() if len(row['post_tokens']) != len(row['consensus_rationale'])]
        if bad:
            df = df.drop(index=bad).reset_index(drop=True)
        return df

    df_test = df_test[df_test['post_tokens'].apply(len) > 0].reset_index(drop=True)
    df_test = verify_alignment(df_test)

    test_words = df_test['post_tokens'].tolist()
    test_rats = df_test['consensus_rationale'].tolist()
    test_labels = [config.LABEL_MAPPING[l] for l in df_test['majority_label']]
    test_targets = df_test['final_target_category'].tolist()

    return test_words, test_rats, test_labels, test_targets

# -------------------------------------------------------------------
# Prediction helper – unchanged
# -------------------------------------------------------------------
def get_predictions(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_token_probs = []
    all_token_valid_mask = []
    all_token_rationale_mask = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_valid_mask = batch['token_valid_mask'].to(device)
            token_rationale_mask = batch['token_rationale_mask'].to(device)

            sequence_logits, token_logits = model(
                input_ids, attention_mask, token_valid_mask
            )

            probs = torch.softmax(sequence_logits, dim=-1)
            preds = torch.argmax(sequence_logits, dim=-1)

            B, L, C = token_logits.shape
            pred_idx = preds.view(B, 1, 1).expand(B, L, 1)
            pred_token_logits = token_logits.gather(dim=2, index=pred_idx).squeeze(-1)
            token_probs = torch.sigmoid(pred_token_logits) * token_valid_mask

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_token_probs.append(token_probs.cpu())
            all_token_valid_mask.append(token_valid_mask.cpu())
            all_token_rationale_mask.append(token_rationale_mask.cpu())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        torch.cat(all_token_probs, dim=0),
        torch.cat(all_token_valid_mask, dim=0),
        torch.cat(all_token_rationale_mask, dim=0)
    )

# -------------------------------------------------------------------
# Explainability utilities (span-based) – unchanged
# -------------------------------------------------------------------
def find_consecutive_spans(binary_mask):
    indices = np.where(binary_mask == 1)[0]
    if len(indices) == 0:
        return []
    spans = []
    for k, g in groupby(enumerate(indices), lambda x: x[1] - x[0]):
        group = list(g)
        spans.append((group[0][1], group[-1][1] + 1))
    return spans

def compute_span_iou_f1(model_spans, human_spans, iou_threshold=0.5):
    if not model_spans and not human_spans:
        return 1.0, 1.0, 1.0
    if not model_spans or not human_spans:
        return 0.0, 0.0, 0.0
    def iou(s1, s2):
        start1, end1 = s1
        start2, end2 = s2
        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = (end1 - start1) + (end2 - start2) - intersection
        return intersection / union if union > 0 else 0.0
    matched_pred = sum(1 for m in model_spans if any(iou(m, h) >= iou_threshold for h in human_spans))
    matched_human = sum(1 for h in human_spans if any(iou(h, m) >= iou_threshold for m in model_spans))
    prec = matched_pred / len(model_spans) if model_spans else 0.0
    rec = matched_human / len(human_spans) if human_spans else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

# -------------------------------------------------------------------
# Explainability metrics computation (adapted for CAP) – unchanged
# -------------------------------------------------------------------
def compute_explainability_metrics(model, test_loader, device, tokenizer,
                                   preds, labels, probs,
                                   token_probs, token_valid_mask, token_rationale_mask):
    """
    Computes token-level and span-level explainability metrics,
    as well as comprehensiveness and sufficiency, using direct token predictions.
    token_probs are already probabilities (sigmoid of predicted-class token logits).
    """
    print("\n" + "="*80)
    print("Computing Explainability Metrics (CAP model)")
    print("="*80)

    # token_probs, valid_mask, rat_mask are already tensors, we'll convert to numpy
    token_probs_np = token_probs.numpy()
    valid_mask_np = token_valid_mask.numpy()
    rat_mask_np = token_rationale_mask.numpy()

    token_precisions, token_recalls, token_f1s = [], [], []
    iou_scores = []
    span_precs, span_recs, span_f1s = [], [], []
    all_probs_for_auprc, all_human = [], []  # for AUPRC

    toxic_count = 0

    for i in range(len(preds)):
        if labels[i] == 0 or rat_mask_np[i].sum() == 0:
            continue
        toxic_count += 1

        pred_binary = (token_probs_np[i] > 0.5).astype(float) * valid_mask_np[i]
        gold = rat_mask_np[i] * valid_mask_np[i]
        valid_bool = valid_mask_np[i].astype(bool)

        p_mask = pred_binary[valid_bool].astype(bool)
        g_mask = gold[valid_bool].astype(bool)

        tp = (p_mask & g_mask).sum()
        fp = (p_mask & ~g_mask).sum()
        fn = (~p_mask & g_mask).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        token_precisions.append(prec)
        token_recalls.append(rec)
        token_f1s.append(f1)
        intersection = (p_mask & g_mask).sum()
        union = (p_mask | g_mask).sum()
        iou = intersection / union if union > 0 else 0.0
        iou_scores.append(iou)

        # Span-based IoU
        model_spans = find_consecutive_spans(pred_binary)
        human_spans = find_consecutive_spans(gold)
        sp, sr, sf = compute_span_iou_f1(model_spans, human_spans)
        span_precs.append(sp)
        span_recs.append(sr)
        span_f1s.append(sf)

        # AUPRC: collect all token-level probabilities (valid tokens only)
        for j in range(len(valid_bool)):
            if valid_bool[j]:
                all_probs_for_auprc.append(token_probs_np[i, j])
                all_human.append(gold[j])

    # Faithfulness: comprehensiveness & sufficiency
    comprehensiveness_scores, sufficiency_scores = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Faithfulness"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            token_valid_mask_batch = batch['token_valid_mask'].to(device)
            token_rationale_mask_batch = batch['token_rationale_mask'].to(device)

            # Recompute predictions for this batch
            seq_logits, token_logits = model(input_ids, attention_mask, token_valid_mask_batch)
            probs_batch = torch.softmax(seq_logits, dim=-1)
            preds_batch = torch.argmax(seq_logits, dim=-1)

            # Get token probabilities for predicted class
            B, L, C = token_logits.shape
            pred_idx_batch = preds_batch.view(B, 1, 1).expand(B, L, 1)
            pred_token_logits_batch = token_logits.gather(dim=2, index=pred_idx_batch).squeeze(-1)
            tok_probs_batch = torch.sigmoid(pred_token_logits_batch) * token_valid_mask_batch

            for i in range(len(labels_batch)):
                if labels_batch[i].item() == 0 or token_rationale_mask_batch[i].sum() == 0:
                    continue
                orig_prob = probs_batch[i, labels_batch[i]].item()
                pred_tok_binary = (tok_probs_batch[i] > 0.5).float() * token_valid_mask_batch[i]
                M_set = set(torch.where(pred_tok_binary == 1)[0].tolist())

                # Comprehensiveness: mask those tokens
                masked_ids = input_ids[i].clone()
                for t_idx in M_set:
                    if t_idx < len(masked_ids):
                        masked_ids[t_idx] = tokenizer.mask_token_id
                masked_out = model(masked_ids.unsqueeze(0),
                                   attention_mask[i].unsqueeze(0),
                                   token_valid_mask_batch[i].unsqueeze(0))
                masked_prob = torch.softmax(masked_out[0], dim=-1)[0, labels_batch[i]].item()
                comprehensiveness_scores.append(orig_prob - masked_prob)

                # Sufficiency: keep only those tokens
                suff_ids = torch.full_like(input_ids[i], tokenizer.pad_token_id)
                suff_att = torch.zeros_like(attention_mask[i])
                suff_ids[0] = tokenizer.cls_token_id
                suff_att[0] = 1
                last_idx = 0
                for t_idx in M_set:
                    if t_idx < len(suff_ids):
                        suff_ids[t_idx] = input_ids[i][t_idx]
                        suff_att[t_idx] = 1
                        if t_idx > last_idx:
                            last_idx = t_idx
                if last_idx + 1 < len(suff_ids):
                    suff_ids[last_idx + 1] = tokenizer.sep_token_id
                    suff_att[last_idx + 1] = 1

                # Create a new token_valid_mask for sufficiency: only CLS, kept tokens, SEP
                suff_token_valid = torch.zeros_like(token_valid_mask_batch[i])
                suff_token_valid[0] = 1.0
                for t_idx in M_set:
                    if t_idx < len(suff_token_valid):
                        suff_token_valid[t_idx] = 1.0
                if last_idx + 1 < len(suff_token_valid):
                    suff_token_valid[last_idx + 1] = 1.0

                suff_out = model(suff_ids.unsqueeze(0), suff_att.unsqueeze(0), suff_token_valid.unsqueeze(0))
                suff_prob = torch.softmax(suff_out[0], dim=-1)[0, labels_batch[i]].item()
                sufficiency_scores.append(orig_prob - suff_prob)

    if toxic_count == 0:
        print("No toxic examples with rationales found.")
        return None

    auprc = average_precision_score(all_human, all_probs_for_auprc) if all_human else 0.0

    results = {
        'plausibility': {
            'token_precision': np.mean(token_precisions) if token_precisions else 0.0,
            'token_recall': np.mean(token_recalls) if token_recalls else 0.0,
            'token_f1': np.mean(token_f1s) if token_f1s else 0.0,
            'iou': np.mean(iou_scores) if iou_scores else 0.0,
            'span_iou_precision': np.mean(span_precs) if span_precs else 0.0,
            'span_iou_recall': np.mean(span_recs) if span_recs else 0.0,
            'span_iou_f1': np.mean(span_f1s) if span_f1s else 0.0,
            'auprc': auprc,
        },
        'faithfulness': {
            'comprehensiveness': np.mean(comprehensiveness_scores) if comprehensiveness_scores else 0.0,
            'sufficiency': np.mean(sufficiency_scores) if sufficiency_scores else 0.0,
        },
        'num_examples': toxic_count
    }

    print(f"\nExamples evaluated: {toxic_count}")
    print("\nPlausibility:")
    print(f"  Token F1:          {results['plausibility']['token_f1']:.3f}")
    print(f"  Token IOU:         {results['plausibility']['iou']:.3f}")
    print(f"  Span IOU F1:       {results['plausibility']['span_iou_f1']:.3f}")
    print(f"  AUPRC:             {results['plausibility']['auprc']:.3f}")
    print("\nFaithfulness:")
    print(f"  Comprehensiveness: {results['faithfulness']['comprehensiveness']:.3f}")
    print(f"  Sufficiency:       {results['faithfulness']['sufficiency']:.3f}")
    return results

# -------------------------------------------------------------------
# Bias metrics – unchanged
# -------------------------------------------------------------------
def compute_bias_metrics(model, test_loader, device, target_categories):
    print("\n" + "="*80)
    print("Computing Bias Metrics (HateXplain Method)")
    print("="*80)
    model.eval()
    selected = ['African', 'Islam', 'Jewish', 'Homosexual', 'Women',
                'Refugee', 'Arab', 'Caucasian', 'Asian', 'Hispanic']

    all_labels, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Bias evaluation")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_valid_mask = batch['token_valid_mask'].to(device)

            # Bias only needs classification probabilities
            sequence_logits, _ = model(input_ids, attention_mask, token_valid_mask)
            probs = torch.softmax(sequence_logits, dim=-1)
            toxic_probs = probs[:, 1:].sum(dim=1).cpu().numpy()

            for i in range(len(labels)):
                # index in target_categories: batch_idx * batch_size + i
                data_idx = batch_idx * test_loader.batch_size + i
                all_labels.append(1 if labels[i].item() > 0 else 0)
                all_probs.append(toxic_probs[i])
                all_targets.append(target_categories[data_idx])

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    community_metrics = {}
    for comm in selected:
        mask = np.array([comm in (t or []) for t in all_targets])
        if mask.sum() == 0:
            continue
        sub_labels = all_labels[mask]
        sub_probs = all_probs[mask]
        subgroup_auc = roc_auc_score(sub_labels, sub_probs) if len(np.unique(sub_labels)) > 1 else 0.5

        bpsn_mask = (mask & (all_labels == 0)) | (~mask & (all_labels == 1))
        bpsn_labels = all_labels[bpsn_mask]
        bpsn_probs = all_probs[bpsn_mask]
        bpsn_auc = roc_auc_score(bpsn_labels, bpsn_probs) if bpsn_mask.sum() > 0 and len(np.unique(bpsn_labels)) > 1 else 0.5

        bnsp_mask = (mask & (all_labels == 1)) | (~mask & (all_labels == 0))
        bnsp_labels = all_labels[bnsp_mask]
        bnsp_probs = all_probs[bnsp_mask]
        bnsp_auc = roc_auc_score(bnsp_labels, bnsp_probs) if bnsp_mask.sum() > 0 and len(np.unique(bnsp_labels)) > 1 else 0.5

        community_metrics[comm] = {
            'mentions': int(mask.sum()),
            'subgroup_auc': float(subgroup_auc),
            'bpsn_auc': float(bpsn_auc),
            'bnsp_auc': float(bnsp_auc),
        }

    p = -5
    valid = [c for c in community_metrics if community_metrics[c]['mentions'] > 0]
    if valid:
        gmb_sub = np.power(np.mean(np.power([community_metrics[c]['subgroup_auc'] for c in valid], p)), 1/p)
        gmb_bpsn = np.power(np.mean(np.power([community_metrics[c]['bpsn_auc'] for c in valid], p)), 1/p)
        gmb_bnsp = np.power(np.mean(np.power([community_metrics[c]['bnsp_auc'] for c in valid], p)), 1/p)
    else:
        gmb_sub = gmb_bpsn = gmb_bnsp = 0.5

    overall_auc = roc_auc_score(all_labels, all_probs)

    results = {
        'overall_auc': float(overall_auc),
        'communities': community_metrics,
        'gmb_metrics': {
            'gmb_subgroup_auc': float(gmb_sub),
            'gmb_bpsn_auc': float(gmb_bpsn),
            'gmb_bnsp_auc': float(gmb_bnsp),
        },
        'n_communities_analyzed': len(valid),
    }
    print(f"\nOverall AUC: {overall_auc:.4f}")
    for comm, m in sorted(community_metrics.items(), key=lambda x: x[1]['mentions'], reverse=True):
        print(f"{comm:<15} {m['mentions']:<8} {m['subgroup_auc']:<12.4f} {m['bpsn_auc']:<12.4f} {m['bnsp_auc']:<12.4f}")
    print(f"\nGMB (p={p}): Subgroup={gmb_sub:.4f}  BPSN={gmb_bpsn:.4f}  BNSP={gmb_bnsp:.4f}")
    return results

# -------------------------------------------------------------------
# Error analysis – unchanged
# -------------------------------------------------------------------
def save_error_cases(predictions, labels, probabilities, texts, output_file="error_cases.json"):
    label_names = ['Normal', 'Offensive', 'Hate speech']
    error_cases = []
    for idx in range(len(predictions)):
        pred, true = int(predictions[idx]), int(labels[idx])
        if pred != true:
            prob = [float(x) for x in probabilities[idx]]
            error_cases.append({
                "text": texts[idx],
                "true_label": label_names[true],
                "predicted_label": label_names[pred],
                "confidence": float(max(prob)),
                "predicted_probs": {"normal": prob[0], "offensive": prob[1], "hate_speech": prob[2]},
                "error_type": f"{label_names[true]}_as_{label_names[pred]}"
            })
    with open(output_file, "w") as f:
        json.dump(error_cases, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(error_cases)} error cases to {output_file}")

# -------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate CAP Model from Hugging Face Hub")
    parser.add_argument('--model_id', type=str, default=config.HF_MODEL_ID,
                        help="Hugging Face model identifier (default: anonymous-CAP/Hate_bert_lambda1)")
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--save_results', type=str, default=None)
    parser.add_argument('--no_explainability', action='store_true')
    parser.add_argument('--no_bias', action='store_true')
    args, unknown = parser.parse_known_args()

    device = config.DEVICE
    print(f"Using device: {device}")

    # Load test data
    print("Loading and preparing test split...")
    test_words, test_rats, test_labels, test_targets = load_and_split_data()
    print(f"Test samples: {len(test_words)}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    test_dataset = TokenAlignedDataset(test_words, test_rats, test_labels, tokenizer, max_length=config.MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Load model from Hugging Face Hub
    print(f"Loading model from Hugging Face: {args.model_id}")
    model = CAPModel.from_pretrained(args.model_id)
    model.to(device)
    model.eval()

    # Get predictions and token probabilities
    print("\nRunning classification evaluation...")
    preds, labels, probs, token_probs, token_valid, token_rats = get_predictions(
        model, test_loader, device
    )

    # Classification metrics
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    try:
        labels_bin = label_binarize(labels, classes=[0,1,2])
        auroc = roc_auc_score(labels_bin, probs, average='macro', multi_class='ovr')
    except:
        auroc = 0.5

    print("\nTest Set Performance:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  AUROC:     {auroc:.4f}")
    print("\n", classification_report(labels, preds, target_names=['Normal', 'Offensive', 'Hate speech'], digits=4))

    # Error cases
    texts = [' '.join(tokens) for tokens in test_words]
    save_error_cases(preds, labels, probs, texts)

    # Explainability
    explainability_results = None
    if not args.no_explainability:
        explainability_results = compute_explainability_metrics(
            model, test_loader, device, tokenizer,
            preds, labels, probs,
            token_probs, token_valid, token_rats
        )

    # Bias metrics
    bias_results = None
    if not args.no_bias and any(t is not None for t in test_targets):
        bias_results = compute_bias_metrics(model, test_loader, device, test_targets)
    elif not args.no_bias:
        print("\nSkipping bias metrics: no target categories found.")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"F1: {macro_f1:.4f}  |  Accuracy: {acc:.4f}  |  AUROC: {auroc:.4f}")
    if explainability_results:
        p = explainability_results['plausibility']
        f = explainability_results['faithfulness']
        print(f"Plausibility: Token F1={p['token_f1']:.3f}  Span IOU F1={p['span_iou_f1']:.3f}  AUPRC={p['auprc']:.3f}")
        print(f"Faithfulness: Comp={f['comprehensiveness']:.3f}  Suff={f['sufficiency']:.3f}")
    if bias_results:
        g = bias_results['gmb_metrics']
        print(f"Bias: Overall AUC={bias_results['overall_auc']:.4f}  GMB Sub={g['gmb_subgroup_auc']:.4f}  "
              f"BPSN={g['gmb_bpsn_auc']:.4f}  BNSP={g['gmb_bnsp_auc']:.4f}")

    if args.save_results:
        save_data = {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'precision': precision,
            'recall': recall,
            'auroc': auroc,
            'predictions': preds.tolist(),
            'labels': labels.tolist(),
            'probabilities': probs.tolist(),
        }
        if explainability_results:
            save_data['explainability'] = explainability_results
        if bias_results:
            save_data['bias'] = bias_results
        with open(args.save_results, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to {args.save_results}")

if __name__ == "__main__":
    main()
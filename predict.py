import gc
import itertools
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from joblib import dump, load
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import OneClassSVM
from tqdm.auto import tqdm

from model import llm_model, llm_tokenizer, max_length


class Entropy:
    @staticmethod
    def compute_entropy(input_ids, logits, attention_mask):
        with torch.no_grad():
            # Compute log softmax of logits
            logits = torch.log_softmax(logits.float(), dim=-1)
            
            # Tokens (excluding first token) and attention mask (excluding first token)
            tokens = input_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]
            
            # Logits for next token prediction (exclude last logit as it predicts beyond input)
            next_logits = logits[:, :-1, :]
            
            # Entropy of the distribution (D) and log likelihood (L) of the actual token
            entD = torch.sum(next_logits * torch.exp(next_logits), dim=-1)
            entL = torch.gather(next_logits, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
            
            # Apply attention mask to ignore padding
            entD = -torch.where(attention_mask.bool(), entD, torch.tensor(float('nan')))
            entL = -torch.where(attention_mask.bool(), entL, torch.tensor(float('nan')))
        
        return entD.cpu().numpy(), entL.cpu().numpy()


class Batch:
    def __init__(self, iterable, size=1):
        self.iterable = iterable
        self.size = size
        self.len = len(range(0, len(self.iterable), self.size))
    
    def __iter__(self):
        l_batch = len(self.iterable)
        n = self.size
        for mini in range(0, l_batch, n):
            yield self.iterable[mini: min(mini + n, l_batch)]
    
    def __len__(self):
        return self.len


def feature_extraction(tab, batch_size, llm_model, llm_tokenizer, max_length):
    feats_list = ['Dmed', 'Lmed', 'Dp05', 'Lstd', 'meanchr']
    
    # Ensure model is on the correct device
    device = next(llm_model.parameters()).device
    
    for index_list in tqdm(Batch(tab.index, batch_size)):
        texts = [tab.loc[index, 'text'] for index in index_list]
        
        # Tokenize inputs
        tokens = llm_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        # Move tensors to the correct device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Forward pass through the model
        logits = llm_model(**tokens).logits
        
        # Compute entropy
        vetD, vetL = Entropy.compute_entropy(tokens['input_ids'], logits, tokens['attention_mask'])
        
        # Compute features
        tab.loc[index_list, 'meanchr'] = tab.loc[index_list, 'len_chr'].values / np.sum(np.isfinite(vetL), axis=-1)
        tab.loc[index_list, 'Dmed'] = np.nanmedian(vetD, axis=-1)
        tab.loc[index_list, 'Lmed'] = np.nanmedian(vetL, axis=-1)
        tab.loc[index_list, 'Dp05'] = np.nanpercentile(vetD, 5, axis=-1)
        tab.loc[index_list, 'Lstd'] = np.nanstd(vetL, axis=-1)
    
    return tab, feats_list


if __name__ == "__main__":
    # Set training file
    train_csv = 'data/train_essays.csv' if len(sys.argv) < 2 else sys.argv[1]
    
    # Load training data
    train_tab = pd.read_csv(train_csv)
    
    # Add character length column
    train_tab['len_chr'] = train_tab['text'].apply(len)
    
    # Feature extraction
    batch_size = 3
    train_tab, feats_list = feature_extraction(train_tab, batch_size, llm_model, llm_tokenizer, max_length)
    
    # Take only features of real data
    train_feats = train_tab[train_tab['generated'] == 0][feats_list].values
    
    # Z-score normalization
    z_mean = np.mean(train_feats, axis=0, keepdims=True)
    z_std = np.maximum(np.std(train_feats, axis=0, keepdims=True), 1e-4)
    train_feats = (train_feats - z_mean) / z_std
    np.savez('zscore.npz', z_std=z_std, z_mean=z_mean)
    
    # Train classifier
    classifier = OneClassSVM(verbose=1, kernel='rbf', gamma='scale', nu=0.05)
    classifier.fit(train_feats)
    dump(classifier, 'oneClassSVM.joblib')
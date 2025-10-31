# extract_logits_main.py

import os
import json
import torch
import numpy as np
import pandas as pd
from models import MLPClassifier
from utils import compute_OvR_AUC
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Embedding-based Dataset -----
class EmbeddingDataset(Dataset):
    def __init__(self, df, cases_dict):
        case_to_idx = {row["CaseName"]: i for i, row in df.iterrows()}

        self.case_names = []
        self.data = []
        self.labels = []

        for case_name, label in cases_dict.items():  
            if case_name not in case_to_idx:
                print(f"Warning: {case_name} not found in embeddings file — skipping.")
                continue

            idx = case_to_idx[case_name]
            embedding = df.iloc[idx][1:].values.astype(np.float32)
            self.data.append(embedding)
            self.labels.append(label)
            self.case_names.append(case_name)

        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]

        if label not in [0, 1, 2]:
            print(f"DEBUG — BAD LABEL DETECTED in EMBEDDING dataset at index {idx}, label={label}")
        assert label in [0, 1, 2], f"Found invalid label {label} — please check EMBEDDING split files."

        return x, label


# ----- Load full embeddings -----
embeddings_df = pd.read_csv("./embeddings/renal_embeddings.csv")

# ----- Configurable -----
BASE_SAVE = "./results"
DATA_SPLIT_ROOT = "./splits"  # where json files live: data-*/val_split_x.json, data-*/test_split.json


for fold_test in range(5):

    # ----------- TEST (OUTER) -----------
    test_json = os.path.join(DATA_SPLIT_ROOT, f"data-{fold_test}/test_split.json")
    with open(test_json, "r") as f:
        test_cases = json.load(f)

    test_dataset = EmbeddingDataset(embeddings_df, test_cases)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model_path = f"{BASE_SAVE}/test_{fold_test}/models/model.pt"
    model = MLPClassifier(dim_in=512, out_dim=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    probs, true_labels = [], []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = torch.softmax(model(data), dim=1).detach().cpu().numpy()
            probs.append(outputs)
            true_labels.extend(labels.numpy())

    probs = np.vstack(probs)

    df_test = pd.DataFrame({
        "CaseName": test_dataset.case_names,
        "True": true_labels,
        "prob0": probs[:, 0],
        "prob1": probs[:, 1],
        "prob2": probs[:, 2],
    })
    df_test.to_csv(f"{BASE_SAVE}/test_{fold_test}.csv", index=False)

    # ----------- VALIDATION (INNER) -----------
    for fold_inner in range(5):
        val_json = os.path.join(DATA_SPLIT_ROOT, f"data-{fold_test}/val_split_{fold_inner}.json")
        if not os.path.exists(val_json):
            continue

        with open(val_json, "r") as f:
            val_cases = json.load(f)

        val_dataset = EmbeddingDataset(embeddings_df, val_cases)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model_path = f"{BASE_SAVE}/test_{fold_test}/fold_{fold_inner}/models/model.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        probs, true_labels = [], []

        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                outputs = torch.softmax(model(data), dim=1).detach().cpu().numpy()
                probs.append(outputs)
                true_labels.extend(labels.numpy())

        probs = np.vstack(probs)

        df_val = pd.DataFrame({
            "CaseName": val_dataset.case_names,
            "True": true_labels,
            "prob0": probs[:, 0],
            "prob1": probs[:, 1],
            "prob2": probs[:, 2],
        })
        df_val.to_csv(f"{BASE_SAVE}/test_{fold_test}_val_{fold_inner}.csv", index=False)
print("Logits extraction completed for all folds.")

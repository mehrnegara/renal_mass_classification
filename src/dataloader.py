import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import h5py
import numpy as np
import torchvision.transforms.functional as TF


# ----- Embedding-based Dataset -----
class EmbeddingDataset(Dataset):
    def __init__(self, df, cases_dict):
        case_to_idx = {row["CaseName"]: i for i, row in df.iterrows()}

        self.data = []
        self.labels = []

        for case_name, label in cases_dict.items():  
            idx = case_to_idx[case_name]
            embedding = df.iloc[idx][1:].values.astype(np.float32)

            self.data.append(embedding)
            self.labels.append(label)  

        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.long)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
        return x, label


# ----- Image-based Dataset -----
class ImageDataset(Dataset):
    def __init__(self, cases_dict, data_path, slice_index=[69]):
        self.case_names = list(cases_dict.keys())
        self.labels = torch.tensor([cases_dict[c] for c in self.case_names])
        self.data_path = data_path
        self.slice_index = slice_index

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        case = self.case_names[idx]
        path = self.data_path + case + ".h5"

        with h5py.File(path, "r") as hdf:
            arterial = torch.from_numpy(np.asarray(hdf["arterial"]))[:,:,self.slice_index].squeeze().unsqueeze(0)
            venous   = torch.from_numpy(np.asarray(hdf["venous"]))[:,:,self.slice_index].squeeze().unsqueeze(0)
            delay    = torch.from_numpy(np.asarray(hdf["delay"]))[:,:,self.slice_index].squeeze().unsqueeze(0)
            precon   = torch.from_numpy(np.asarray(hdf["precon"]))[:,:,self.slice_index].squeeze().unsqueeze(0)

            arterial = TF.pad(arterial, (0, 10, 0, 11))
            venous   = TF.pad(venous,   (0, 10, 0, 11))
            delay    = TF.pad(delay,    (0, 10, 0, 11))
            precon   = TF.pad(precon,   (0, 10, 0, 11))

            combined = torch.cat((arterial, venous, delay, precon), 0)

        label = self.labels[idx]
        return combined.float(), label


# ----- Factory Loader -----
def create_dataloader(cfg, split_json_path, batch_size, shuffle=True):
    mode = cfg["data"]["mode"]

    # Load labels first
    with open(split_json_path, "r") as f:
        labels_dict = json.load(f)

    # Embedding Mode
    if mode == "embeddings":
        embeddings_df = pd.read_csv(cfg["data"]["embeddings_path"])
        dataset = EmbeddingDataset(embeddings_df, labels_dict)

    # Raw Image Mode
    elif mode == "images":
        dataset = ImageDataset(
            cases_dict=labels_dict,
            data_path=cfg["data"]["data_path"],
            slice_index=[69]  # fixed as confirmed
        )

    else:
        raise ValueError(f"Unsupported data.mode: {mode}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

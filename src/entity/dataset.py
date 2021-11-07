'''
Author: your name
Date: 2021-10-22 15:55:02
LastEditTime: 2021-11-07 11:57:09
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /mdap/src/entity/others/dataset.py
'''
import torch

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


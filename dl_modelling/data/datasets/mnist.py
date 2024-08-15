import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd


class MNIST(Dataset):
    def __init__(self, csv_fl=None, df=None):
        self.csv_fl = csv_fl
        self.df = df
        self.transforms = None
        self.load()

    def load(self):
        if self.csv_fl is not None:
            self.df = pd.read_csv(self.csv_fl)

    def __len__(self):
        if self.df is not None:
            return len(self.df)
        else:
            return 0
        
    def __getitem__(self, idx):
        lbl = self.df.iloc[idx]["label"]
        img = self.df.iloc[idx][1:].values.reshape(28,28)
        if self.transforms is not None:
            return self.transforms(img), lbl
        else:
            return img, lbl


class COND_MNIST(MNIST):
    def __init__(self, num, csv_fl=None, df=None):
        super().__init__(csv_fl, df)
        self.cls = num
        self.df = self.df[self.df.loc[:, "label"] == self.cls]


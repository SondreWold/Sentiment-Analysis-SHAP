import pandas as pd
from torch.utils.data import Dataset
import torch


class NorecDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path, sep=',')
        df.columns = ["Sentiment", "Document"]
        self.documents = list(df['Document'])
        self.labels = list(df['Sentiment'])

    def __getitem__(self, idx):
        return (self.documents[idx], torch.LongTensor([self.labels[idx]]))

    def __len__(self):
        return len(self.documents)

import torch.utils.data as data
import pandas as pd


class ShopeeDataset(data.Dataset):
    def __init__(self, data_set: pd.DataFrame):
        self._data = data_set.title.values
        self._label = data_set.product_id.values

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx], self._label[idx]

import numpy as np
import torch.utils.data as data
import pandas as pd
from sentencepiece import SentencePieceProcessor


class ShopeeDataset(data.Dataset):
    def __init__(self, data_set: pd.DataFrame,
                 tokenizer: SentencePieceProcessor,
                 padding=120,
                 to_lowercase=True):
        self._data = data_set.title.values
        self._label = data_set.product_id.values

        self._tokenizer: SentencePieceProcessor = tokenizer
        self._padding = padding
        self._to_lowercase = to_lowercase

    def __len__(self):
        return len(self._data)

    def get_labels(self):
        return self._label

    def pad(self, tokens):
        pad_id = self._tokenizer.pad_id()
        padded = tokens[:self._padding]
        padded = [pad_id] * (self._padding - len(padded)) + padded
        return padded

    def __getitem__(self, idx):
        title = self._data[idx]
        if self._to_lowercase:
            title = title.lower()
        tokens = self._tokenizer.EncodeAsIds(title)
        tokens = self.pad(tokens)
        return np.array(tokens), np.array(self._label[idx])

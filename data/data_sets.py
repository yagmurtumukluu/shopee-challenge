from config import SHOPEE_CSV
import pandas as pd
import numpy as np


def unique_token_ration(title: str):
    tokens = title.lower()
    tokens = tokens.split(' ')
    return float(len(set(tokens))) / float(len(tokens))


def get_data_csv(test_split: int):
    df = pd.read_csv(SHOPEE_CSV)
    df['unique_token_ratio'] = df.title.apply(unique_token_ration)
    df = df.loc[df.unique_token_ratio > .5]
    dataset = []
    for product_id, group in df.groupby(by="label_group"):
        if len(group) >= 2:
            for row in group.iterrows():
                sample = {'product_id': product_id,
                          'title': row['title']}
                dataset.append(sample)
    dataset = pd.DataFrame(dataset)
    pids = dataset.product_id.unique()
    test_ids = np.random.choice(pids, test_split)
    test_set = dataset.loc[dataset.product_id.isin(test_ids)]
    train_set = dataset.loc[~dataset.product_id.isin(test_ids)]
    return train_set, test_set


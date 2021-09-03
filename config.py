import os
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

SHOPEE_CSV = os.getenv("SHOPEE_CSV", '/Users/cenk/data-sets/shopee/shopee-product-matching/train.csv')
SPIECE_MODEL_FILE = os.getenv("SPIECE_MODEL_FILE", 'artifacts/spiece.model')
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", 'lstm')


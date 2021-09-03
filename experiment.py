import logging

import pytorch_metric_learning.utils.logging_presets as logging_presets
import sentencepiece as spm
import torch
from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from config import SPIECE_MODEL_FILE, DEVICE, TB_LOGS_PATH
from data.data_sets import get_data_csv
from data.pml_dataset import ShopeeDataset
from dl.lstm import LSTMClassifier
from dl.mlp import MLP

sp = spm.SentencePieceProcessor()
logging.info(sp.load(SPIECE_MODEL_FILE))

train, test = get_data_csv(1000)

train = ShopeeDataset(train, sp)
test = ShopeeDataset(test, sp)

trunk = LSTMClassifier()
embedder = MLP([128, 128])

trunk.to(DEVICE)
embedder.to(DEVICE)

record_keeper, _, _ = logging_presets.get_record_keeper("/tmp/", TB_LOGS_PATH)

loss = losses.CircleLoss()

# Set the mining function
miner = miners.BatchHardMiner()

# Set the dataloader sampler
sampler = samplers.MPerClassSampler(
    train.get_labels(), m=2, length_before_new_iter=len(train)
)

# Set optimizers
trunk_optimizer = torch.optim.Adam(
    trunk.parameters(), lr=0.0001, weight_decay=0.0001
)
embedder_optimizer = torch.optim.Adam(
    embedder.parameters(), lr=0.0001, weight_decay=0.0001
)

# Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {
    "trunk_optimizer": trunk_optimizer,
    "embedder_optimizer": embedder_optimizer,
}
loss_funcs = {"metric_loss": loss}
mining_funcs = {"tuple_miner": miner}

hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"val": test}


# Create the tester
tester = testers.GlobalEmbeddingSpaceTester(
    end_of_testing_hook=hooks.end_of_testing_hook,
    dataloader_num_workers=4,
    accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
)

end_of_epoch_hook = hooks.end_of_epoch_hook(
    tester, dataset_dict, '/tmp', test_interval=1, patience=1
)

trainer = trainers.MetricLossOnly(
    models,
    optimizers,
    32,
    loss_funcs,
    mining_funcs,
    train,
    sampler=sampler,
    dataloader_num_workers=0,
    end_of_iteration_hook=hooks.end_of_iteration_hook,
    end_of_epoch_hook=end_of_epoch_hook,
)
trainer.train(num_epochs=10)

torch.save(models["trunk"].state_dict(), '/tmp' + "/trunk.pth")
torch.save(models["embedder"].state_dict(), '/tmp' + "/embedder.pth")

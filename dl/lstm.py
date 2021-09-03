import torch
import torch.nn as nn
from dl.layers import SpatialDropout


class LSTMClassifier(nn.Module):
    def __init__(
            self,
            vocab_size=1000,
            embed_size=128,
            output_size=128,
            lstm_units=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=True
    ):
        super(LSTMClassifier, self).__init__()
        self.output_size = output_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding_dropout = SpatialDropout(dropout)
        self.lstm = nn.LSTM(
            embed_size,
            lstm_units,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_last_layer_size = lstm_units * 4 if bidirectional else lstm_units * 2

        self.fc = nn.Linear(lstm_last_layer_size, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()  # coz https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/14
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        outputs, hidden = self.lstm(h_embedding)

        self.lstm.flatten_parameters()
        avg_pool = torch.mean(outputs, 1)
        max_pool, _ = torch.max(outputs, 1)
        out = torch.cat((max_pool, avg_pool), 1)
        out = self.fc(out)
        return out

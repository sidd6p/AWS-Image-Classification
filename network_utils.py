from torch import nn

import torch.nn.functional as F


class MyNetwork(nn.Module):
    def __init__(self, input_size, hidden_units, drop_rate, output_size):
        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units // 2)
        self.layer3 = nn.Linear(hidden_units // 2, output_size)

        self.dropout = nn.Dropout(drop_rate)

        self.output = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.layer2(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.layer3(out)

        out = self.output(out)

        return out

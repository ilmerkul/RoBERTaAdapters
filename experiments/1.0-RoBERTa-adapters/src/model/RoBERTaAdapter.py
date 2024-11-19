import torch.nn as nn


class RoBERTaAdapter(nn.Module):
    def __init__(self, ff_linear, r, h):
        super(RoBERTaAdapter, self).__init__()

        self.ff_linear = ff_linear
        self.lin0 = nn.Linear(h, r)
        self.act = nn.GELU()
        self.lin1 = nn.Linear(r, h)

    def forward(self, hidden_states, input_tensor):
        x = self.ff_linear(hidden_states, input_tensor)
        z = self.lin0(x)
        z = self.act(z)
        z = self.lin1(z)

        z = z + x

        return z

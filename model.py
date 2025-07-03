import torch as t
from torch import nn


class NameGeneratorLSTM(nn.Module):
    def __init__(self, input_size: int = 53, hidden_size: int = 64, output_size: int = 53, layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.rnn = nn.LSTM(input_size, hidden_size, layers)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self,
                x: t.Tensor,
                h_c: tuple[t.Tensor, t.Tensor] = None,
        ) -> t.Tensor:
        if h_c is None:
            h = t.zeros(3, x.size(1), self.hidden_size)
            c = t.zeros(3, x.size(1), self.hidden_size)
            h_c = (h, c)

        out, h_c = self.rnn(x, h_c)
        out = self.fc(out)
        out = out.squeeze(1)                # out: [seq_len, n_letters]        
        return out, h_c

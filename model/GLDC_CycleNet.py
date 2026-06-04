import torch
import torch.nn as nn


class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.d_model = configs.d_model
        self.use_revin = configs.use_norm

        self.use_gldc = 1

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        if self.use_gldc:
            self.cycleQueue_ext = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        input_dim = self.seq_len * 2 if self.use_gldc else self.seq_len

        self.model = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.pred_len)
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None, cycle_index=None):
        # x: [B, L, C]

        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        p_input = self.cycleQueue(cycle_index, self.seq_len)
        x_residual = x - p_input

        if self.use_gldc:
            p_ext = self.cycleQueue_ext(cycle_index, self.seq_len)
            x_in = torch.cat([x_residual, p_ext], dim=1)
        else:
            x_in = x_residual

        y = self.model(x_in.permute(0, 2, 1))
        y = y.permute(0, 2, 1)  # [B, S, C]

        p_output = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
        y = y + p_output

        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y
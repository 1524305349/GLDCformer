import torch
import torch.nn as nn
import torch.nn.functional as F


class moving_avg(nn.Module):

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = 1
        self.cycle_len = configs.cycle

        self.use_seasonal_prior = 1
        self.use_trend_prior = 1

        if self.use_seasonal_prior:
            self.cycleQueue_S = nn.Parameter(torch.zeros(self.cycle_len, self.channels))
        if self.use_trend_prior:
            self.cycleQueue_T = nn.Parameter(torch.zeros(self.cycle_len, self.channels))

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        self.in_dim_S = self.seq_len * 2 if self.use_seasonal_prior else self.seq_len
        self.in_dim_T = self.seq_len * 2 if self.use_trend_prior else self.seq_len

        if self.individual:
            self.Linear_S_Weight = nn.Parameter(torch.Tensor(self.channels, self.in_dim_S, self.pred_len))
            self.Linear_S_Bias = nn.Parameter(torch.Tensor(self.channels, self.pred_len))
            self.Linear_T_Weight = nn.Parameter(torch.Tensor(self.channels, self.in_dim_T, self.pred_len))
            self.Linear_T_Bias = nn.Parameter(torch.Tensor(self.channels, self.pred_len))

            for w in [self.Linear_S_Weight, self.Linear_T_Weight]:
                nn.init.xavier_uniform_(w)
            for b in [self.Linear_S_Bias, self.Linear_T_Bias]:
                nn.init.zeros_(b)
        else:
            self.Linear_Seasonal = nn.Linear(self.in_dim_S, self.pred_len)
            self.Linear_Trend = nn.Linear(self.in_dim_T, self.pred_len)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None, cycle_index=None):
        B, L, C = x.shape

        seasonal_init, trend_init = self.decompsition(x)

        idx_offset = torch.arange(L, device=x.device)
        gather_index = (cycle_index.view(-1, 1) + idx_offset.view(1, -1)) % self.cycle_len

        if self.use_seasonal_prior:
            p_prior_S = self.cycleQueue_S[gather_index]  # [B, L, C]
            seasonal_ext = torch.cat([seasonal_init, p_prior_S], dim=1)  # [B, 2*L, C]
        else:
            seasonal_ext = seasonal_init

        if self.use_trend_prior:
            p_prior_T = self.cycleQueue_T[gather_index]  # [B, L, C]
            trend_ext = torch.cat([trend_init, p_prior_T], dim=1)  # [B, 2*L, C]
        else:
            trend_ext = trend_init

        if self.individual:
            seasonal_ext = seasonal_ext.permute(2, 0, 1)
            trend_ext = trend_ext.permute(2, 0, 1)

            seasonal_output = torch.bmm(seasonal_ext, self.Linear_S_Weight) + self.Linear_S_Bias.unsqueeze(1)
            trend_output = torch.bmm(trend_ext, self.Linear_T_Weight) + self.Linear_T_Bias.unsqueeze(1)

            out = (seasonal_output + trend_output).permute(1, 2, 0)
        else:
            seasonal_ext = seasonal_ext.permute(0, 2, 1)
            trend_ext = trend_ext.permute(0, 2, 1)

            seasonal_output = self.Linear_Seasonal(seasonal_ext)
            trend_output = self.Linear_Trend(trend_ext)
            out = (seasonal_output + trend_output).permute(0, 2, 1)

        return out
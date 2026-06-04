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
        self.cycle_len = configs.cycle
        self.individual = 1

        self.cycleQueue = nn.Parameter(torch.zeros(self.cycle_len, self.channels))

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        combined_len = self.seq_len * 2

        if self.individual:
            self.Linear_Seasonal_Weight = nn.Parameter(torch.Tensor(self.channels, combined_len, self.pred_len))
            self.Linear_Trend_Weight = nn.Parameter(torch.Tensor(self.channels, combined_len, self.pred_len))
            self.Linear_Seasonal_Bias = nn.Parameter(torch.Tensor(self.channels, self.pred_len))
            self.Linear_Trend_Bias = nn.Parameter(torch.Tensor(self.channels, self.pred_len))

            nn.init.xavier_uniform_(self.Linear_Seasonal_Weight)
            nn.init.xavier_uniform_(self.Linear_Trend_Weight)
            nn.init.zeros_(self.Linear_Seasonal_Bias)
            nn.init.zeros_(self.Linear_Trend_Bias)
        else:
            self.Linear_Seasonal = nn.Linear(combined_len, self.pred_len)
            self.Linear_Trend = nn.Linear(combined_len, self.pred_len)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None, cycle_index=None):
        # x: [B, L, C]
        B, L, C = x.shape

        idx_offset = torch.arange(L, device=x.device)
        gather_index = (cycle_index.view(-1, 1) + idx_offset.view(1, -1)) % self.cycle_len
        p_prior = self.cycleQueue[gather_index]  # [B, L, C]

        seasonal_init, trend_init = self.decompsition(x)

        seasonal_ext = torch.cat([seasonal_init, p_prior], dim=1)
        trend_ext = torch.cat([trend_init, p_prior], dim=1)

        if self.individual:
            seasonal_ext = seasonal_ext.permute(2, 0, 1)
            trend_ext = trend_ext.permute(2, 0, 1)

            seasonal_output = torch.bmm(seasonal_ext,
                                        self.Linear_Seasonal_Weight) + self.Linear_Seasonal_Bias.unsqueeze(1)
            trend_output = torch.bmm(trend_ext, self.Linear_Trend_Weight) + self.Linear_Trend_Bias.unsqueeze(1)

            out = (seasonal_output + trend_output).permute(1, 2, 0)
        else:
            seasonal_ext = seasonal_ext.permute(0, 2, 1)
            trend_ext = trend_ext.permute(0, 2, 1)

            seasonal_output = self.Linear_Seasonal(seasonal_ext)
            trend_output = self.Linear_Trend(trend_ext)
            out = (seasonal_output + trend_output).permute(0, 2, 1)

        return out
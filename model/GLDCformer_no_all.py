import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, n_norm=False):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        self.n_norm = n_norm
        if self.n_norm:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if self.n_norm:
            x = self.norm(x)

        ffn_out = self.ffn(x)

        return ffn_out


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, D = x.shape
        H = self.n_heads
        d_k = self.d_k
        q = self.q_linear(x).view(B, C, H, d_k).transpose(1, 2)
        k = self.k_linear(x).view(B, -1, H, d_k).transpose(1, 2)
        v = self.v_linear(x).view(B, -1, H, d_k).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, C, D)
        return self.out_linear(out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, ffn_layers, n_norm, dropout):
        super().__init__()

        self.attn = Attention(d_model, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model) if n_norm else nn.Identity()

        self.ffn_layers = nn.ModuleList([
            FFN(d_model=d_model, d_ff=d_ff, n_norm=n_norm)
            for _ in range(ffn_layers)
        ])

    def forward(self, x):
        res = x
        x = self.attn_norm(x)
        x = self.attn(x) + res

        ffn_res = x
        for ffn in self.ffn_layers:
            x = ffn(x)
        x = x + ffn_res

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.use_norm = configs.use_norm

        self.input_proj = nn.Linear(self.seq_len, configs.d_model)

        self.EncoderLayers = nn.ModuleList([
            EncoderLayer(
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                n_heads=4,
                ffn_layers=configs.ffn_layers,
                n_norm=configs.n_norm,
                dropout=0.5
            ) for _ in range(configs.e_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, self.pred_len)
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None, cycle_index=None):

        if self.use_norm:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x_input = self.input_proj(x.permute(0, 2, 1))

        hidden = x_input
        for layer in self.EncoderLayers:
            hidden = layer(hidden)

        output = self.output_proj(hidden).permute(0, 2, 1)

        if self.use_norm:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import DataEmbedding_inverted


class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, input_size, feature_dim, seq_len, d_model, nhead, dim_feedforward, num_layers, dropout_rate, output_attention=False, use_norm=False):
        super(iTransformer, self).__init__()
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.stem = nn.Linear(input_size, feature_dim)
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(c_in=seq_len, d_model=d_model, dropout=dropout_rate)  # only linear
        # Encoder-only architecture
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        attention=FullAttention(mask_flag=False, attention_dropout=dropout_rate, output_attention=output_attention),
                        d_model=d_model, n_heads=nhead
                    ),
                    d_model=d_model,
                    d_ff=dim_feedforward,
                    dropout=dropout_rate,
                    activation="relu"
                ) for l in range(num_layers)
            ],
            conv_layers=None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.time_projector = nn.Linear(d_model, 1, bias=False)
        self.feature_projector = nn.Linear(feature_dim, 1, bias=False)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.stem(x_enc)  # extra steps
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.time_projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x):  # B L N
        forecast_out = self.forecast(x, None, None, None)  # B S(1) N
        out = self.feature_projector(forecast_out) # B 1 1
        return out.squeeze()  # B
    
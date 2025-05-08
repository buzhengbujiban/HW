from typing import List
from collections import OrderedDict
import math
import torch
import torch.nn as nn
from einops import rearrange


class MLPNetwork(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: List[int], dropout_rate: float, act_fn: str="ELU", bn: bool=False
    ) -> nn.Module:
        super(MLPNetwork, self).__init__()
        # affine, stat = False, False
        layers = []
        input = input_size
        for i, hidden in enumerate(hidden_size):
            layer = []
            layer.append((f"Dense_{(i + 1)}", nn.Linear(input, hidden)))
            if act_fn is not None:
                layer.append((f"{act_fn}_{(i + 1)}", getattr(nn, act_fn)()))
            if bn:
                layer.append((f"BN_{(i + 1)}", nn.BatchNorm1d(hidden)))
            if dropout_rate > 0:
                layer.append((f"Drop_{(i + 1)}", nn.Dropout(p=dropout_rate)))
            input = hidden
            layers.extend(layer)
        self.layer = torch.nn.Sequential(OrderedDict(layers))
        self.output_layer = nn.Linear(input, 1)
        self.get_last_hidden = False

    def forward(self, x):
        x = self.layer(x)
        if self.get_last_hidden:
            return x
        return self.output_layer(x).squeeze()  # shape (N,)

class DenseBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate, act_fn=None, norm_type:str=None, require_linear=True):
        super(DenseBlock, self).__init__()
        # (Linear -> BatchNorm -> Activation -> Dropout)
        block = []
        if require_linear:
            block.append(("linear", nn.Linear(input_size, output_size)))
        if norm_type is not None:
            block.append(("norm", getattr(nn, norm_type)(output_size)))
        if act_fn is not None:
            block.append(("act", getattr(nn, act_fn)()))
        if dropout_rate > 0:
            block.append(("drop", nn.Dropout(dropout_rate)))
        self.block = nn.Sequential(OrderedDict(block))
    def forward(self, x):
        return self.block(x)

class MLP3(nn.Module):
    def __init__(self, input_size: int, hidden_size: List[int], dropout_rate: float, act_fn: str="ELU", norm_type=None, lastlayer_reg: bool=True):
        super(MLP3, self).__init__()
        self.layers = nn.ModuleList()
        input = input_size
        for i, hidden in enumerate(hidden_size):
            if i < len(hidden_size)-1 or lastlayer_reg:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
            else:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=0, act_fn=act_fn, norm_type=None))
            input = hidden
        self.layers = nn.Sequential(*self.layers)
        self.output_layer = nn.Linear(input, 1)
    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x).squeeze()  # shape (N,)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (Max, F) -> (1, Max, F)
        if not self.batch_first:
            pe = pe.transpose(0, 1)  # (1, Max, F) -> (Max, 1, F)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if not self.batch_first:
            # [T, N, F]
            return x + self.pe[: x.size(0), :, :]
        # [N, T, F]
        return x + self.pe[:, : x.size(1), :]


class TransformerNetwork(nn.Module):
    def __init__(
        self, input_size, d_model, dropout_rate, nhead, num_layers, dim_feedforward=None, norm_first=True
    ):
        super(TransformerNetwork, self).__init__()
        self.feature_layer = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, batch_first=True)
        if dim_feedforward is None:
            dim_feedforward = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.decoder_layer = nn.Linear(d_model, 1)
        self.d_feat = input_size

    def forward(self, src):
        # src [N, T, I] --> [N, T, F]
        src = self.feature_layer(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        # [N, T, F] --> [N, 1]
        output = self.decoder_layer(output[:, -1, :])

        return output.squeeze()


class TransformerEncoder(nn.Module):
    """This is basically a decomposed Transformer Encoder layer with output layer"""

    def __init__(
        self, input_size, d_model, dropout_rate, nhead, dim_feedforward=None
    ):
        super(TransformerEncoder, self).__init__()
        self.embed_layer = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, batch_first=True)
        self.attention_layer = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout_rate, batch_first=True
        )
        if dim_feedforward is None:
            dim_feedforward = d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(d_model, 1)  # bias=False

    def forward(self, x, extract_attn=False):
        embeded = self.embed_layer(x)  # (N, T, I) -> (N, T, F)
        embeded = self.pos_encoder(embeded)
        attn_output, attn_output_weights = self.attention_layer(
            embeded, embeded, embeded, need_weights=extract_attn
        )  # (N, T, F); (N, T, T) which .sum(axis=1) = 1 representing each timestamp's weight
        if extract_attn:
            return attn_output_weights
        res1 = self.norm1(
            embeded + self.dropout(attn_output)
        )  # (N, T, F), 1st residual connection with self-attention in middle
        res2 = self.norm2(
            res1
            + self.dropout(
                self.linear2(self.dropout(self.activation(self.linear1(res1))))
            )
        )  # (N, T, F), 2nd residual connection with fully-connected in middle
        output = self.output_layer(
            res2[:, -1, :]
        ).squeeze()  # (N, T, F) -> (N, F) -> (N, 1) -> (N,)
        return output


class RNNNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_rate: float,
        num_layers: int=1,
        type: str="GRU", init_type: str="xavier_normal",
    ) -> nn.Module:
        super(RNNNetwork, self).__init__()

        if type == "RNN":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate,
            )
        elif type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate,
            )
        else:
            raise ValueError("The type %s is not implemented in RNN!" % (type))

        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)  # in case of gradient disappearing
        for name, p in self.rnn.named_parameters():
            if name.startswith("weight"):
                if init_type == "xavier_normal":
                    nn.init.xavier_normal_(p)
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(p)
            else:
                nn.init.zeros_(p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, X):
        out, (hidden_prev) = self.rnn(X)
        out = out[:, -1, :]
        out = self.fc(out).squeeze()
        return out

class MLPNet(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: List[int], dropout_rate: float, act_fn: str = "ReLU", bn: bool = False,
    ):
        super(MLPNet, self).__init__()
        layers = []
        input_dim = input_size
        for i, hidden in enumerate(hidden_size):
            layer = []
            layer.append((f"Dense_{i + 1}", nn.Linear(input_dim, hidden)))
            if bn:
                layer.append((f"BN_{i + 1}", nn.BatchNorm1d(hidden)))
            layer.append((f"{act_fn}_{i + 1}", getattr(nn, act_fn)()))
            layer.append((f"Drop_{i + 1}", nn.Dropout(p=dropout_rate)))
            input_dim = hidden
            layers.extend(layer)
        self.layer = nn.Sequential(OrderedDict(layers))
        self._initialize_weights()

    def forward(self, x):
        return self.layer(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_last_share_layer(self):
        # Retrieve the last fully connected layer
        for layer in reversed(self.layer):
            if isinstance(layer, nn.Linear):
                return layer
        return None

class MultiTargetNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: List[int], output_size: int, dropout_rate: float, act_fn: str = "ReLU", bn: bool = False, 
                 head_hidden_size: List[int]=None):
        super(MultiTargetNetwork, self).__init__()
        self.share_model = MLPNet(input_size, hidden_size, dropout_rate, act_fn, bn)
        self.head_hidden_size = head_hidden_size
        if head_hidden_size is None:
            self.regression_model = nn.Linear(hidden_size[-1], output_size)
        else:
            self.regression_models = nn.ModuleList([
                MLPNetwork(hidden_size[-1], head_hidden_size, dropout_rate, act_fn, bn) for _ in range(output_size)
            ])
            
    def forward(self, x):
        features = self.share_model(x)
        if self.head_hidden_size is None:
            predictions = self.regression_model(features)
        else:
            predictions = []
            for i in range(len(self.regression_models)):
                predictions.append(self.regression_models[i](features))
            predictions = torch.stack(predictions, dim=1)
        return predictions
    
class SingleTargetNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: List[int], dropout_rate: float, act_fn: str = "ReLU", bn: bool = False, 
                 head_hidden_size: List[int]=None):
        super(SingleTargetNetwork, self).__init__()
        self.share_model = MLPNet(input_size, hidden_size, dropout_rate, act_fn, bn)
        self.head_hidden_size = head_hidden_size
        if head_hidden_size is None:
            self.regression_model = nn.Linear(hidden_size[-1], 1)
        else:
            self.regression_model = MLPNetwork(hidden_size[-1], head_hidden_size, dropout_rate, act_fn, bn)
            
    def forward(self, x):
        features = self.share_model(x)
        predictions = self.regression_model(features).squeeze()
        return predictions
from typing import List, Dict, Tuple
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from qrfitter3.nnfitter.network_zoo.network import MLPNetwork, DenseBlock, MLP3
from pattern_helper import *
from einops import rearrange



class SimpleTSMLP(nn.Module):
    
    def __init__(
        self, seq_len: int, input_size: int, hidden_size: List[int], dropout_rate: float, act_fn: str="ELU", bn: bool=False
    ):
        super(SimpleTSMLP, self).__init__()
        self.mlp_network = MLPNetwork(input_size, hidden_size, dropout_rate, act_fn, bn)
        self.ts_projector = nn.Linear(seq_len, 1)

    def forward(self, x):
        x = self.ts_projector(x.permute(0, 2, 1)).squeeze()  # B L N -> B N L -> B N 1 -> B N
        out = self.mlp_network(x).squeeze()  # -> B
        return out

class TSMLP(nn.Module):
    def __init__(
        self, input_size: int, seq_len: int, d_hidden: int, d_time_pattern: int, dropout_rate: float=0.3, act_fn: str="GELU", norm_type:str=None
    ):
        super(TSMLP, self).__init__()
        self.embed_layer = nn.Linear(input_size, d_hidden)
        self.time_pattern_extraction_layer = nn.Sequential(
            # nn.LayerNorm(d_hidden),
            nn.Linear(seq_len, d_time_pattern),
            getattr(nn, act_fn)(),
            nn.Dropout(dropout_rate)
        )
        self.time_pattern_output_layer = nn.Sequential(
            nn.Linear(d_time_pattern * d_hidden, d_hidden),
            getattr(nn, act_fn)(),
            nn.Dropout(dropout_rate)
        )
        self.mlp_layers = nn.Sequential(
                # nn.LayerNorm(d_hidden),
                nn.Linear(input_size, d_hidden),
                getattr(nn, act_fn)(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_hidden, d_hidden),
                getattr(nn, act_fn)(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_hidden, d_hidden),
                getattr(nn, act_fn)()
        )
        self.output_layer = nn.Linear(d_hidden, 1, bias=False)
    
    def forward(self, x):
        x_mlp_out = self.mlp_layers(x[:, -1, :])  # B T F -> B F -> B H
        x_embed = self.embed_layer(x)  # B T F -> B T H
        x_time_pattern = self.time_pattern_extraction_layer(x_embed.permute(0, 2, 1))  # B T H -> B H T -> B H P
        x_time_pattern_out = self.time_pattern_output_layer(torch.flatten(x_time_pattern, start_dim=1))  # B H P -> B H*P -> B H
        out = self.output_layer(x_mlp_out+x_time_pattern_out).squeeze()  # B H -> B
        return out

class SkipConnectionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, act_fn, norm_type):
        super(SkipConnectionMLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        self.layers.append(DenseBlock(input_size, hidden_size[0], dropout_rate, act_fn, norm_type=norm_type))
        
        # Hidden layers with skip connections
        for i in range(1, len(hidden_size)):
            self.layers.append(DenseBlock(hidden_size[i-1], hidden_size[i], dropout_rate, act_fn, norm_type=norm_type))
            self.skip_connections.append(nn.Linear(input_size, hidden_size[i]))  # Skip connection from input
        
        # Output layer (Linear only)
        self.output_layer = nn.Linear(hidden_size[-1], 1, bias=False)
        
    def forward(self, x):
        input_x = x  # Save input for skip connections
        # First layer
        x = self.layers[0](x)
        
        # Hidden layers with skip connections
        for i in range(1, len(self.layers)):
            x = self.layers[i](x) + self.skip_connections[i-1](input_x)
        
        # Output layer
        x = self.output_layer(x).squeeze()  # shape (N,)
        return x


class IdentityBlock(nn.Module):
    def forward(self, x):
        return x


class ResiConnectionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, act_fn, norm_type:str=None, num_hidden: int|None=None, lastlayer_reg: bool=True, firstpurelinear: bool=False):
        super(ResiConnectionMLP, self).__init__()
        if isinstance(num_hidden, int):
            assert isinstance(hidden_size, int)
            hidden_size = [hidden_size] * num_hidden
        assert isinstance(hidden_size, list)
        self.layers = nn.ModuleList()
        self.resi_connections = nn.ModuleList()

        if firstpurelinear:
            self.layers.append(DenseBlock(input_size, hidden_size[0], dropout_rate=0, act_fn=None, norm_type=None))  # 改成纯线性层
        else:
            self.layers.append(DenseBlock(input_size, hidden_size[0], dropout_rate, act_fn=act_fn, norm_type=norm_type))
        
        # Hidden layers with skip connections
        for i in range(1, len(hidden_size)):
            if i < len(hidden_size)-1 or lastlayer_reg:
                self.layers.append(DenseBlock(hidden_size[i-1], hidden_size[i], dropout_rate, act_fn=act_fn, norm_type=norm_type))
            else:
                self.layers.append(DenseBlock(hidden_size[i-1], hidden_size[i], dropout_rate=0, act_fn=act_fn, norm_type=None))
            if hidden_size[i-1] != hidden_size[i]:
                self.resi_connections.append(nn.Linear(hidden_size[i-1], hidden_size[i]))  # Skip connection from input
            else:
                self.resi_connections.append(IdentityBlock())
        # Output layer (Linear only)
        self.output_layer = nn.Linear(hidden_size[-1], 1, bias=False)
        
    def forward(self, x):
        # First layer
        x = self.layers[0](x)
        
        # Hidden layers with skip connections
        for i in range(1, len(self.layers)):
            x = self.layers[i](x) + self.resi_connections[i-1](x)
        
        # Output layer
        x = self.output_layer(x).squeeze()  # shape (N,)
        return x


class GroupInputMLP(nn.Module):
    def __init__(self, input_size, x_pattern, reduce_factor, hidden_size, dropout_rate=0.3, act_fn="ELU", norm_type=None):
        super(GroupInputMLP, self).__init__()
        terms = read_columns_from_pattern(x_pattern)  # "/data/beef2/junming/data/combine1999/terms.YYYYMMDD.sdiv"
        groups = sorted(list(set([v[:3] for v in terms])))
        self.group_index = {}
        first_hidden_size = 0
        for g in groups:
            g_terms = [v for v in terms if v.startswith(g)]
            start_idx, end_idx = terms.index(g_terms[0]), terms.index(g_terms[-1])
            assert end_idx - start_idx + 1 == len(g_terms), f"terms in {g} are not continuous"
            self.group_index[g] = (start_idx, end_idx)
            g_input_size = len(g_terms)
            g_hidden_size = g_input_size // reduce_factor
            setattr(self, f"{g}_input", DenseBlock(g_input_size, g_hidden_size, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
            first_hidden_size += g_hidden_size
        
        layers = []
        input = first_hidden_size
        for hidden in hidden_size:
            layers.append(DenseBlock(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
            input = hidden
        layers.append(DenseBlock(input, 1, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        first_hidden = []
        for g, (start_idx, end_idx) in self.group_index.items():
            g_x = x[:, start_idx: end_idx+1]
            g_hidden = getattr(self, f"{g}_input")(g_x)
            first_hidden.append(g_hidden)
        first_hidden = torch.cat(first_hidden, dim=1)
        out = self.layers(first_hidden).squeeze(dim=1)
        return out


def determine_hidden_size(cluster_id, feature_num, hidden_mode=0, num_hidden=3, normal_group_hidden=5, unique_group_multiplier=4):
    if hidden_mode == 0:
        hidden_size = [math.ceil(feature_num / 4), math.ceil(feature_num / 8), math.ceil(feature_num / 16)]  # 暂时先hardcode 3层
    else:  
        if cluster_id == -1:
            last_hidden = normal_group_hidden * unique_group_multiplier
        else:
            last_hidden = normal_group_hidden
        if hidden_mode == 1:
            hidden_size = [math.ceil(feature_num * 0.2)] * (num_hidden-1)  # adaptive_last_hidden=False
        elif hidden_mode == 2:
            hidden_size = [256, 64]
        elif hidden_mode == 3:
            if feature_num >= 256:
                hidden_size = [256, 64]
            elif feature_num >= 128:
                hidden_size = [128, 32]
            elif feature_num >= 64:
                hidden_size = [64, 16]
            else:
                hidden_size = [32, 8]
        hidden_size.append(last_hidden)
    return hidden_size

class GroupHeadMLP(nn.Module):
    def __init__(
            self, input_size, cluster_nums: List[Tuple[int, int]], hidden_mode=1, num_hidden=3, normal_group_hidden=5, unique_group_multiplier=4, dropout_rate=0.3, act_fn="ELU", norm_type=None,
            freeze_head=False, init_adjust_var=True
        ):
        super(GroupHeadMLP, self).__init__()
        # cluster_num中的顺序应当与forward中输入的x的顺序一样，通过指定use_x_names做到
        last_hidden_sum = 0
        self.cluster_nums = cluster_nums
        for cluster_id, feature_num in self.cluster_nums:
            hidden_size = determine_hidden_size(cluster_id=cluster_id, feature_num=feature_num, hidden_mode=hidden_mode, num_hidden=num_hidden, normal_group_hidden=normal_group_hidden, unique_group_multiplier=unique_group_multiplier)
            last_hidden_sum += hidden_size[-1]
            head = FeedForward(sizes=[feature_num]+hidden_size, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type, lastlayer_reg=False)  # 最后一层隐层之后没有dropout
            setattr(self, f"head_{cluster_id}", head)
        self.output_layer = nn.Linear(last_hidden_sum, 1, bias=False)
        self.freeze_head = freeze_head
        self.init_adjust_var = init_adjust_var

    def init_after_load_share(self):
        if self.freeze_head:  # init完head通过这个参数决定要不要freeze
            for cluster_id, feature_num in self.cluster_nums:
                head = getattr(self, f"head_{cluster_id}")
                for param in head.parameters():
                    param.requires_grad = False

        cluster_id, _ = self.cluster_nums[0]
        init_output_layer = hasattr(self, f"output_layer_{cluster_id}")
        if init_output_layer:  # 如果有f"output_layer_{cluster_id}"，那么代表各submodel的output_layer已经在learner处被load了进来
            cluster_output_layers = []
            for cluster_id, _ in self.cluster_nums:
                cluster_output_layer = getattr(self, f"output_layer_{cluster_id}")
                cluster_output_layers.append(cluster_output_layer)
                delattr(self, f"output_layer_{cluster_id}")
            num_cluster = len(self.cluster_nums)
            num_output_layer_in = sum([cluster_output_layer.in_features for cluster_output_layer in cluster_output_layers])
            output_layer = nn.Linear(num_output_layer_in, 1, bias=False)
            weight = torch.cat([cluster_output_layer.weight for cluster_output_layer in cluster_output_layers], dim=1)
            if self.init_adjust_var:
                weight /= np.sqrt(num_cluster)
            output_layer.weight = nn.Parameter(weight)
            self.output_layer = output_layer

    def forward(self, x):
        start_idx = 0
        head_outs = []
        for cluster_id, feature_num in self.cluster_nums:
            cluster_x = x[:, start_idx: start_idx+feature_num]
            head = getattr(self, f"head_{cluster_id}")
            # print(f"cluster_id: {cluster_id}; cluster_x.shape: {cluster_x.shape}; head: {head}")
            head_out = head(cluster_x)
            # print(f"head_out: {head_out.shape}")
            head_outs.append(head_out)
            start_idx += feature_num

        head_outs = torch.cat(head_outs, dim=1)
        # print(f"head_outs: {head_outs.shape}; self.output_layer: {self.output_layer}")
        out = self.output_layer(head_outs).squeeze(dim=1)
        return out

class SingleGroupMLP(nn.Module):
    def __init__(self, input_size, cluster_num: Tuple[int, int], hidden_mode=1, num_hidden=3, normal_group_hidden=5, unique_group_multiplier=4, dropout_rate=0.3, act_fn="ELU", norm_type=None):
        super(SingleGroupMLP, self).__init__()
        cluster_id, feature_num = cluster_num
        assert input_size==feature_num
        hidden_size = determine_hidden_size(cluster_id=cluster_id, feature_num=feature_num, hidden_mode=hidden_mode, num_hidden=num_hidden, normal_group_hidden=normal_group_hidden, unique_group_multiplier=unique_group_multiplier)
        self.head = FeedForward(sizes=[feature_num]+hidden_size, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type, lastlayer_reg=False)  # 最后一层隐层之后没有dropout
        self.output_layer = nn.Linear(hidden_size[-1], 1, bias=False)
    def forward(self, x):
        head_out = self.head(x)
        out = self.output_layer(head_out).squeeze(dim=1)
        return out

class Autoencoder(nn.Module):

    def __init__(self, input_size:int, hidden_size:List[int], bottleneck_dim:int, dropout_rate=0., act_fn="ELU", norm_type=None, self_gating=False, gate_act_fn='Sigmoid', resnet=False):
        super(Autoencoder, self).__init__()

        encoder_layers = []
        input = input_size
        for hidden in hidden_size:
            encoder_layers.append(DenseBlockNEW(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type, self_gating=self_gating, gate_act_fn=gate_act_fn, resnet=resnet))
            input = hidden
        encoder_layers.append(DenseBlockNEW(input, bottleneck_dim, dropout_rate=0, act_fn=act_fn, norm_type=None))  # only linear, add act_fn
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        input = bottleneck_dim
        for hidden in hidden_size[::-1]:
            decoder_layers.append(DenseBlockNEW(input, hidden, dropout_rate=0, act_fn=act_fn, norm_type=None, self_gating=self_gating, gate_act_fn=gate_act_fn, resnet=resnet))
            input = hidden
        decoder_layers.append(DenseBlockNEW(input, input_size, dropout_rate=0, act_fn=None, norm_type=None, self_gating=self_gating, gate_act_fn=gate_act_fn, resnet=resnet))  # only linear
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


class ConcatHiddenMLP(nn.Module):
    def __init__(self, input_size:int, concat_size:int, hidden_size:List[int], dropout_rate=0.3, act_fn="ELU", norm_type=None):
        super(ConcatHiddenMLP, self).__init__()
        self.concat_size = concat_size  # 可以直接concat到最后一层，放在terms的前面
        self.ffn_size = input_size - concat_size
        layers = []
        input = self.ffn_size
        for hidden in hidden_size:
            layers.append(DenseBlock(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
            input = hidden
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(self.concat_size+input, 1)
        
    def forward(self, x):
        concat_x = x[:, :self.concat_size]
        ffn_x = x[:, self.concat_size:]
        ffn_x = self.layers(ffn_x)
        last_hidden = torch.cat([concat_x, ffn_x], dim=1)
        out = self.output_layer(last_hidden).squeeze(dim=1)
        return out

class EmbeddingMLP(nn.Module):
    def __init__(
            self, input_size:int, embedding_input_size:int, hidden_size:List[int], embedding_loc: int=0, embedding_method: str="add", embedding_size: int=None,
            dropout_rate=0.3, act_fn="ELU", norm_type=None
        ):
        super(EmbeddingMLP, self).__init__()
        self.embedding_input_size = embedding_input_size  # 作为embedding的terms, 放在terms的后面
        self.ffn_size = input_size - embedding_input_size
        self.embedding_loc = embedding_loc
        self.embedding_method = embedding_method
        self.layers = nn.ModuleList()
        assert embedding_loc < len(hidden_size), f"Invalid embedding_loc: {embedding_loc}"
        if embedding_method == "concat":  # 可以concat在任意一层，取决于embedding_loc
            self.embedding_layer = nn.Linear(embedding_input_size, embedding_size)
        if embedding_method in ["add", "multiply"] and embedding_loc == -1:  # add/multiply在input层
            self.embedding_layer = nn.Linear(embedding_input_size, self.ffn_size)
        if embedding_method == "concat" and embedding_loc == -1:  # concat在input层
            input = self.ffn_size + embedding_size
        else:
            input = self.ffn_size
        for i, hidden in enumerate(hidden_size):
            self.layers.append(DenseBlock(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
            input = hidden
            if embedding_loc == i:
                if embedding_method in ["add", "multiply"]:  # add/multiply在隐层
                    self.embedding_layer = nn.Linear(embedding_input_size, hidden)
                else:
                    input = hidden + embedding_size
        self.output_layer = nn.Linear(input, 1)

    def forward(self, x):
        embed_x = x[:, self.ffn_size:]
        ffn_x = x[:, :self.ffn_size]
        embedding = self.embedding_layer(embed_x)
        # print("start", embed_x.shape, ffn_x.shape, embedding.shape)
        if self.embedding_loc == -1:
            if self.embedding_method == "concat":
                ffn_x = torch.cat([embedding, ffn_x], dim=1)
            elif self.embedding_method == "add":
                ffn_x = embedding + ffn_x
            elif self.embedding_method == "multiply":
                ffn_x = embedding * ffn_x
        for i, layer in enumerate(self.layers):
            ffn_x = layer(ffn_x)
            # print("i_start", ffn_x.shape)
            if self.embedding_loc == i:
                if self.embedding_method == "concat":
                    ffn_x = torch.cat([embedding, ffn_x], dim=1)
                elif self.embedding_method == "add":
                    ffn_x = embedding + ffn_x
                elif self.embedding_method == "multiply":
                    ffn_x = embedding * ffn_x
            # print("i_end", ffn_x.shape)
        out = self.output_layer(ffn_x).squeeze(dim=1)
        return out


class MLPSplitInput(nn.Module):
    def __init__(self, input_size: int, group_nums: List[Tuple[str, int]], hidden_size: List[int], dropout_rate: float, act_fn: str="ELU", norm_type=None, lastlayer_reg: bool=True, **kwargs):
        super(MLPSplitInput, self).__init__()
        assert input_size == sum([group_num[1] for group_num in group_nums])
        self.group_nums = group_nums
        for group_num in self.group_nums:
            name, num = group_num
            setattr(self, name, nn.Linear(num, hidden_size[0]))
        self.layers = nn.ModuleList()
        input = input_size
        for i, hidden in enumerate(hidden_size):
            if i == 0:  # 第一个dense已经不需要linear，因为已经有了
                self.layers.append(DenseBlock(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type, require_linear=False))
            elif i < len(hidden_size)-1 or lastlayer_reg:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
            else:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=0, act_fn=act_fn, norm_type=None))
            input = hidden
        self.layers = nn.Sequential(*self.layers)
        self.output_layer = nn.Linear(input, 1, bias=True)

    def forward(self, x):
        out = 0
        start = 0
        for group_num in self.group_nums:
            name, num = group_num
            linear = getattr(self, name)
            end = start + num
            out += linear(x[:, start: end])
            start = end
        x = self.layers(out)
        return self.output_layer(x).squeeze()  # shape (N,)


class FiLMMLP(nn.Module):
    def __init__(
            self, input_size:int, embedding_input_size:int, hidden_size:List[int], embedding_loc: int=0, film_num_hidden: int=0,
            dropout_rate=0.3, act_fn="ELU", norm_type=None
        ):
        super(FiLMMLP, self).__init__()
        self.embedding_input_size = embedding_input_size  # 作为embedding的terms, 放在terms的后面
        self.ffn_size = input_size - embedding_input_size  # terms数目
        self.embedding_loc = embedding_loc

        if embedding_loc == -1:  # 确定FiLM generator network的output size(i.e., the shifting and scaling coefficients)
            embedding_size = self.ffn_size
        else:
            embedding_size = hidden_size[embedding_loc]
        self.gamma_layers, self.beta_layers = nn.ModuleList(), nn.ModuleList()
        embedding_input = embedding_input_size
        for i in range(film_num_hidden):  # 往FiLM generator network中加隐层(如有)
            self.gamma_layers.append(DenseBlock(embedding_input, embedding_size, dropout_rate=0, act_fn=act_fn, norm_type=None))
            self.beta_layers.append(DenseBlock(embedding_input, embedding_size, dropout_rate=0, act_fn=act_fn, norm_type=None))
            embedding_input = embedding_size
        self.gamma_layers.append(nn.Linear(embedding_input, embedding_size))  # 输出层是一个纯Linear
        self.beta_layers.append(nn.Linear(embedding_input, embedding_size))
        self.gamma_layers, self.beta_layers = nn.Sequential(*self.gamma_layers), nn.Sequential(*self.beta_layers)

        self.layers = nn.ModuleList()
        input = self.ffn_size
        for i, hidden in enumerate(hidden_size):
            self.layers.append(DenseBlock(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
            input = hidden
        self.output_layer = nn.Linear(input, 1)
    
    def forward(self, x):
        embed_x = x[:, self.ffn_size:]
        ffn_x = x[:, :self.ffn_size]
        gamma = self.gamma_layers(embed_x)
        beta = self.beta_layers(embed_x)
        if self.embedding_loc == -1:
            ffn_x = gamma * ffn_x + beta
        for i, layer in enumerate(self.layers):
            ffn_x = layer(ffn_x)
            # print("i_start", ffn_x.shape)
            if self.embedding_loc == i:
                ffn_x = gamma * ffn_x + beta
            # print("i_end", ffn_x.shape)
        out = self.output_layer(ffn_x).squeeze(dim=1)
        return out

class FeedForward(nn.Module):  # 就是一个mlp
    def __init__(self, sizes, dropout_rate, act_fn=None, norm_type=None, lastlayer_reg: bool=True):
        super(FeedForward, self).__init__()
        self.layers = nn.ModuleList()
        input = sizes[0]
        for i, hidden in enumerate(sizes[1:]):
            if i < len(sizes[1:])-1 or lastlayer_reg:
                self.layers.append(DenseBlock(input, hidden, dropout_rate, act_fn=act_fn, norm_type=norm_type))
            else:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=0, act_fn=act_fn, norm_type=None))
            input = hidden
        self.layers = nn.Sequential(*self.layers)
    def forward(self, x):
        return self.layers(x)

class ExternalAttention(nn.Module):
    def __init__(
        self, input_size: int, external_input_size: int, d_model: int, dropout_rate, nhead, dim_feedforward:List[int], num_layers=1, act_fn="ELU", norm_type=None, **kwargs
    ):
        super(ExternalAttention, self).__init__()
        self.external_input_size = external_input_size  # external的terms, 放在terms的后面
        self.ffn_size = input_size - external_input_size  # terms数目
        self.ffn_proj_layer = nn.Linear(self.ffn_size, d_model)  # MultiheadAttention里第一步就是将qkv都project到embed_dim，所以这个in_project其实只是为了对齐shape来加residual connect
        self.query_proj_layer = nn.Linear(self.external_input_size, d_model)  # MultiheadAttention里embed_dim必须等于query的dim, 各自会project到embed_dim
        self.norm = None if norm_type is None else getattr(nn, norm_type)(d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout_rate, batch_first=True),
                FeedForward(sizes=[d_model]+dim_feedforward+[d_model], dropout_rate=dropout_rate, act_fn=act_fn, norm_type=None)
            ]))
        self.output_layer = nn.Linear(d_model, 1)
    def forward(self, x):
        # x = x.squeeze(dim=0)
        #print(f"x: {x.shape}")
        external_x = x[:, self.ffn_size:]
        ffn_x = x[:, :self.ffn_size]
        #print(f"external_x: {external_x.shape}")
        #print(f"ffn_x: {ffn_x.shape}")
        qk = self.query_proj_layer(external_x)
        v = self.ffn_proj_layer(ffn_x)
        #print(f"qk: {q.shape} {k.shape}")
        #print(f"v: {v.shape}")
        for attn, ff in self.layers:
            attn_output, _ = attn(qk, qk, v, need_weights=False)
            if self.norm is not None:
                attn_output = self.norm(attn_output)
            #print(f"attn_output: {attn_output.shape}")
            v = attn_output + v
            #print(f"ff(v): {ff(v).shape}")
            v = ff(v) + v
        out = self.output_layer(v).squeeze(dim=1)
        return out


class MLPMT(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: List[int], dropout_rate: float, act_fn: str="ELU", norm_type=None, lastlayer_reg: bool=True, **kwargs):
        super(MLPMT, self).__init__()
        self.layers = nn.ModuleList()
        input = input_size
        for i, hidden in enumerate(hidden_size):
            if i < len(hidden_size)-1 or lastlayer_reg:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
            else:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=0, act_fn=act_fn, norm_type=None))
            input = hidden
        self.layers = nn.Sequential(*self.layers)
        self.output_layer = nn.Linear(input, output_size)
    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)  # shape (N, Y)
    

class FiLMResnet(nn.Module):
    def __init__(
            self, input_size:int, embedding_input_size:int, hidden_size,
            embedding_loc: int=0, film_num_hidden: int=0,
            dropout_rate=0.3, act_fn="ELU", norm_type:str=None,
            num_hidden: int|None=None, lastlayer_reg: bool=False,
            gamma: bool=True, beta: bool=True, tanh: bool=False
        ):
        super(FiLMResnet, self).__init__()
        if isinstance(num_hidden, int):
            assert isinstance(hidden_size, int)
            hidden_size = [hidden_size] * num_hidden
        assert isinstance(hidden_size, list)

        self.embedding_input_size = embedding_input_size  # 作为embedding的terms, 放在terms的后面
        self.ffn_size = input_size - embedding_input_size  # terms数目
        self.embedding_loc = embedding_loc
        if embedding_loc == -1:  # 确定FiLM generator network的output size(i.e., the shifting and scaling coefficients)
            embedding_size = self.ffn_size
        else:
            embedding_size = hidden_size[embedding_loc]
        self.gamma, self.beta = gamma, beta
        self.tanh = tanh
        if gamma:
            self.gamma_layers = nn.ModuleList()
            embedding_input = embedding_input_size
            for i in range(film_num_hidden):  # 往FiLM generator network中加隐层(如有)
                self.gamma_layers.append(DenseBlock(embedding_input, embedding_size, dropout_rate=0, act_fn=act_fn, norm_type=None))
                embedding_input = embedding_size
            self.gamma_layers.append(nn.Linear(embedding_input, embedding_size))  # 输出层是一个纯Linear
            self.gamma_layers = nn.Sequential(*self.gamma_layers)
        if beta:
            self.beta_layers = nn.ModuleList()
            embedding_input = embedding_input_size
            for i in range(film_num_hidden):  # 往FiLM generator network中加隐层(如有)
                self.beta_layers.append(DenseBlock(embedding_input, embedding_size, dropout_rate=0, act_fn=act_fn, norm_type=None))
                embedding_input = embedding_size
            self.beta_layers.append(nn.Linear(embedding_input, embedding_size))
            self.beta_layers = nn.Sequential(*self.beta_layers)

        self.layers = nn.ModuleList()
        self.layers.append(DenseBlock(self.ffn_size, hidden_size[0], dropout_rate=0, act_fn=None, norm_type=None))  # 改成纯线性层
        self.resi_connections = nn.ModuleList()

        # Hidden layers with skip connections
        for i in range(1, len(hidden_size)):
            if i < len(hidden_size)-1 or lastlayer_reg:
                self.layers.append(DenseBlock(hidden_size[i-1], hidden_size[i], dropout_rate, act_fn=act_fn, norm_type=norm_type))
            else:
                self.layers.append(DenseBlock(hidden_size[i-1], hidden_size[i], dropout_rate=0, act_fn=act_fn, norm_type=None))
            if hidden_size[i-1] != hidden_size[i]:
                self.resi_connections.append(nn.Linear(hidden_size[i-1], hidden_size[i]))  # Skip connection from input
            else:
                self.resi_connections.append(IdentityBlock())
        # Output layer (Linear only)
        self.output_layer = nn.Linear(hidden_size[-1], 1, bias=False)
        
    def forward(self, x):
        embed_x = x[:, self.ffn_size:]
        ffn_x = x[:, :self.ffn_size]
        if self.gamma:
            gamma = self.gamma_layers(embed_x)
            if self.tanh:
                gamma = 1 + torch.tanh(gamma)
        else:
            gamma = 1  
        beta = self.beta_layers(embed_x) if self.beta else 0
        if self.embedding_loc == -1:
            ffn_x = gamma * ffn_x + beta
        # First layer
        i = 0
        ffn_x = self.layers[0](ffn_x)
        if self.embedding_loc == i:
            ffn_x = gamma * ffn_x + beta
        # Hidden layers with skip connections
        for i in range(1, len(self.layers)):
            ffn_x = self.layers[i](ffn_x) + self.resi_connections[i-1](ffn_x)
            if self.embedding_loc == i:
                ffn_x = gamma * ffn_x + beta
        # Output layer
        out = self.output_layer(ffn_x).squeeze()  # shape (N,)
        return out


class Expert_Gate(nn.Module):
    def __init__(self, input_size, expert_hidden, n_expert, n_task, use_gate, dropout_rate, act_fn, norm_type): #feature_dim:输入数据的维数  expert_dim:每个神经元输出的维数  n_expert:专家数量  n_task:任务数(gate数)  use_gate：是否使用门控，如果不使用则各个专家取平均
        super(Expert_Gate, self).__init__()
        self.n_task = n_task
        self.use_gate = use_gate
        
        '''专家网络'''
        self.experts = nn.ModuleList()
        for i in range(n_expert):
            expert = FeedForward(sizes=[input_size]+expert_hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type)
            self.experts.append(expert)
        '''门控网络'''
        self.gates = nn.ModuleList()
        for i in range(n_task):
            gate = nn.Sequential(nn.Linear(input_size, n_expert), nn.Softmax(dim=1))
            self.gates.append(gate)
        
    def forward(self, x):
        # expert_dim 即 expert_hidden[-1]
        E_net = [expert(x) for expert in self.experts]  # 元素shape (bs,expert_dim), 长度n_expert
        if self.use_gate:
            # 构建多个专家网络
            E_net = torch.cat(([e[:,np.newaxis,:] for e in E_net]),dim = 1) # 维度 (bs,n_expert,expert_dim)
            # 构建多个门网络
            gate_net = [gate(x) for gate in self.gates]     # 维度 n_task个(bs,n_expert)
            # towers计算：对应的门网络乘上所有的专家网络
            tower_inputs = []
            for i in range(self.n_task):
                g = gate_net[i].unsqueeze(2)  # 维度(bs,n_expert,1)
                tower_i = torch.matmul(E_net.transpose(1,2),g)  # 维度 (bs,expert_dim,1)
                tower_inputs.append(tower_i.transpose(1,2).squeeze(1))  # 维度(bs,expert_dim)
        else:
            tower_inputs = sum(E_net)/len(E_net)  # 每一个tower的输入都将会一样
        return tower_inputs

class MMoE(nn.Module):
    def __init__(
        self,
        input_size, output_size, n_expert, expert_hidden, tower_hidden, use_gate=True,
        dropout_rate: float=0.3, act_fn: str="ELU", norm_type=None, lastlayer_reg: bool=True
    ): 
        super(MMoE, self).__init__()
        
        self.use_gate = use_gate
        self.Expert_Gate = Expert_Gate(
            input_size=input_size, expert_hidden=expert_hidden, n_expert=n_expert, n_task=output_size, use_gate=use_gate,
            dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type
        )
        
        self.towers = nn.ModuleList()
        for i in range(output_size):
            tower = FeedForward(sizes=expert_hidden[-1:]+tower_hidden+[1], dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type)
            self.towers.append(tower)
        
    def forward(self, x):
        
        tower_inputs = self.Expert_Gate(x)
        outs = []
        for i, tower in enumerate(self.towers):
            if self.use_gate:   
                out = tower(tower_inputs[i])
            else:
                out = tower(tower_inputs)
            outs.append(out)
        out = torch.cat(outs, dim = 1)
        return out  # shape (N, Y)


class DynamicLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(DynamicLinear, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, output_size)])
    def get_total_input_size(self):
        return sum([linear.in_features for linear in self.linears])
    def get_output_size(self):
        return self.linears[0].out_features
    def freeze_linears(self):
        for linear in self.linears:
            for para in linear.parameters():
                para.requires_grad = False
    def forward(self, x):
        out = 0
        start = 0
        for linear in self.linears:
            end = start + linear.in_features
            out += linear(x[:, start: end])
            start = end
        return out

class MLPSequential(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: List[int], dropout_rate: float, act_fn: str="ELU", bn: bool=False, lastlayer_reg: bool=False,
        init_0: bool=True, freeze_former_linears: bool=False
    ) -> nn.Module:
        super(MLPSequential, self).__init__()
        self.input_size = input_size
        self.init_0 = init_0
        self.freeze_former_linears = freeze_former_linears
        layers = []
        input = input_size
        for i, hidden in enumerate(hidden_size):
            layer = []
            if i == 0:
                layer.append((f"Dense_{(i + 1)}", DynamicLinear(input, hidden)))
            else:
                layer.append((f"Dense_{(i + 1)}", nn.Linear(input, hidden)))
            if (i < len(hidden_size)-1 or lastlayer_reg) and bn:
                layer.append((f"BN_{(i + 1)}", nn.BatchNorm1d(hidden)))
            if act_fn is not None:
                layer.append((f"actfn_{(i + 1)}", getattr(nn, act_fn)()))
            if (i < len(hidden_size)-1 or lastlayer_reg) and dropout_rate > 0:
                layer.append((f"Drop_{(i + 1)}", nn.Dropout(p=dropout_rate)))
            input = hidden
            layers.extend(layer)
        layers.append(("out", nn.Linear(input, 1, bias=False)))
        self.share_model = torch.nn.Sequential(OrderedDict(layers))

    def init_after_load_share(self):
        dlinear = self.share_model.Dense_1
        assert self.input_size > dlinear.get_total_input_size(), f"self.input_size({self.input_size}) must > dlinear({dlinear.get_total_input_size()})"
        add_linear = nn.Linear(self.input_size - dlinear.get_total_input_size(), dlinear.get_output_size(), bias=False)
        if self.init_0:
            nn.init.constant_(add_linear.weight, 0)
        if self.freeze_former_linears:
            dlinear.freeze_linears()
        dlinear.linears.append(add_linear)
        assert self.input_size == dlinear.get_total_input_size(), f"input_sizes don't match: {self.input_size} != {dlinear.get_total_input_size()}"

    def forward(self, x):
        x = self.share_model(x).squeeze()  # shape (N,)
        return x


class MLP3_guopeng(nn.Module):
    def __init__(self, input_size: int, hidden_size: List[int], dropout_rate: float, act_fn: str="ELU", norm_type=None, lastlayer_reg: bool=True, add_shortcut: bool=False):
        super(MLP3_guopeng, self).__init__()
        self.layers = nn.ModuleList()
        input_size -= 1  # 最后一个col是y
        input = input_size
        for i, hidden in enumerate(hidden_size):
            if i < len(hidden_size)-1 or lastlayer_reg:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
            else:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=0, act_fn=act_fn, norm_type=None))
            input = hidden
        self.layers = nn.Sequential(*self.layers)
        self.add_shortcut = add_shortcut
        if add_shortcut:
            self.shortcut_layer = nn.Linear(input_size, input)
    def forward(self, x):
        x, y = x[:, :-1], x[:, -1]
        if self.add_shortcut:
            x = self.layers(x) + self.shortcut_layer(x)
        else:
            x = self.layers(x) # (B, F)
        # print(f"{x.shape=}, {x.T.shape=}, {torch.inverse(x.T @ x).shape=}")
        f = torch.inverse(x.T @ x) @ x.T @ y  # (F,)
        yhat = x @ f  # (B,)
        return yhat
    

class CustomAffineLayer(nn.Module):
    def __init__(self, input_dim, expand_num=2, act_fn="ReLU"):
        super(CustomAffineLayer, self).__init__()
        self.input_dim = input_dim
        self.expand_num = expand_num
        self.a = nn.Parameter(torch.randn(expand_num, input_dim))
        self.b = nn.Parameter(torch.zeros(expand_num, input_dim))
        if act_fn is not None:
            self.act_fn = getattr(nn, act_fn)()

    def forward(self, x):
        assert x.shape[1] == self.input_dim, "Input dimension mismatch"  # B, F
        outputs = []
        for i in range(self.expand_num):
            out = self.a[i] * x + self.b[i]  # B, F
            if hasattr(self, "act_fn"):
                out = self.act_fn(out)
            outputs.append(out)
        return torch.cat(outputs, dim=1)  # B, nF


class MLP3_addAffineInput(nn.Module):
    def __init__(self, input_size: int, hidden_size: List[int], dropout_rate: float, act_fn: str="ELU", norm_type=None,
                 lastlayer_reg: bool=True, expand_num=2, affine_act_fn="ReLU"):
        super(MLP3_addAffineInput, self).__init__()
        self.affine_input_layer = CustomAffineLayer(input_dim=input_size, expand_num=expand_num, act_fn=affine_act_fn)
        self.layers = nn.ModuleList()
        input = input_size * expand_num
        for i, hidden in enumerate(hidden_size):
            if i < len(hidden_size)-1 or lastlayer_reg:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
            else:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=0, act_fn=act_fn, norm_type=None))
            input = hidden
        self.layers = nn.Sequential(*self.layers)
        self.output_layer = nn.Linear(input, 1)
    def forward(self, x):
        x = self.affine_input_layer(x)
        x = self.layers(x)
        return self.output_layer(x).squeeze()  # shape (N,)
    
class MLP3_addGaussian(nn.Module):
    def __init__(self, std=1, std_min=None, std_max=None, **kwargs):
        super(MLP3_addGaussian, self).__init__()
        if std_min is not None and std_max is not None:
            std = torch.tensor(np.linspace(std_min, std_max, kwargs["input_size"]), dtype=torch.float32).unsqueeze(0)
            self.register_buffer('std', std)
        else:
            self.std = std
        self.mlp = MLP3(**kwargs)
        print(self.std)

    def forward(self, x):
        if self.training:  # 仅在训练模式下添加噪声
            noise = torch.randn_like(x) * self.std
            x += noise
        return self.mlp(x)

class GatedMul(nn.Module):
    def __init__(
            self, input_size: int, hidden_size: List[int], dropout_rate: float, act_fn: str, norm_type: str, lastlayer_reg: bool,
            embedding_input_size:int, embedding_hidden_size, embedding_dropout_rate: float, embedding_act_fn: str, embedding_norm_type: str, embedding_resnet: bool,
            gate_act_fn='Sigmoid', flatten=True):
        super(GatedMul, self).__init__()
        
        self.flatten = flatten
        self.embedding_input_size = embedding_input_size  # 作为embedding的terms, 放在terms的后面
        self.ffn_size = input_size - embedding_input_size  # terms数目
        self.layers = nn.ModuleList()
        input = self.ffn_size
        for i, hidden in enumerate(hidden_size):
            if i < len(hidden_size)-1 or lastlayer_reg:
                self.layers.append(DenseBlockNEW(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
            else:
                self.layers.append(DenseBlockNEW(input, hidden, dropout_rate=0, act_fn=act_fn, norm_type=None))
            input = hidden
        self.layers = nn.Sequential(*self.layers)

        self.gate_layers = nn.ModuleList()
        embedding_input = self.embedding_input_size
        for hidden in embedding_hidden_size:
            self.gate_layers.append(DenseBlockNEW(embedding_input, hidden, dropout_rate=embedding_dropout_rate, act_fn=embedding_act_fn, norm_type=embedding_norm_type, resnet=embedding_resnet))
            embedding_input = hidden
        self.gate_layers = nn.Sequential(*self.gate_layers)
        self.gate_act_fn = getattr(nn, gate_act_fn)()
        
        if flatten:
            self.output_layer = nn.Linear(input * embedding_input, 1)
        else:
            assert input == embedding_input  # 各自的last hidden
            self.output_layer = nn.Linear(input, 1)
        
    def forward(self, x):
        ffn_x = x[:, :self.ffn_size]  # (B, N)
        ffn_out = self.layers(ffn_x)
        embed_x = x[:, self.ffn_size:]  # (B, M)
        gate = self.gate_act_fn(self.gate_layers(embed_x))
        if self.flatten:
            batch_size = x.shape[0]
            out = torch.bmm(gate.unsqueeze(2), ffn_out.unsqueeze(1))  # (B, M, N)
            out = out.view(batch_size, -1)  # (B, MN)
        else:
            out = ffn_out * gate
        return self.output_layer(out).squeeze()  # shape (B,)

class LinearRegression(nn.Module):
    def __init__(self, input_size, bias=True):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=bias)
    def forward(self, x):
        return self.linear(x).squeeze()  # shape (B,)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0., norm_type="LayerNorm"):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by num_heads"
        self.dropout = nn.Dropout(dropout_rate)
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_linear = DenseBlock(embed_dim, embed_dim, dropout_rate=0, act_fn=None, norm_type=norm_type)

    def forward(self, x):
        # x (b, t, f)
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # tuple((b, t, f), (b, t, f), (b, t, f))
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h = self.num_heads), qkv)  # (b, h, t, d) each
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (b, h, t, t)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)  # (b, h, t, t)
        out = torch.matmul(attention, v) # (b, h, t, d)
        out = rearrange(out, 'b h t d -> b t (h d)')  # (b, t, f)
        return self.out_linear(out)  # (b, t, f)
    
class Attention2d(nn.Module):
    def __init__(self, embed_dim, dropout_rate=0., norm_type="LayerNorm"):
        super(Attention2d, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_linear = DenseBlock(embed_dim, embed_dim, dropout_rate=0, act_fn=None, norm_type=norm_type)

    def forward(self, x):
        # x (b, f)
        b, f = x.size()
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)  # tuple((b, f), (b, f), (b, f))
        scores = torch.matmul(k.transpose(-1, -2), q) / (b ** 0.5)  # (f, f)  # 这里是K.T @ Q的目的是想得到的score_{ij}是Corr(K_{fi}, Q_{fj})
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)  # (f, f)
        out = torch.matmul(v, attention) # (b, f)  # 按照scores的含义，这样是用单个q对所有k的corr来加权v
        return self.out_linear(out)  # (b, f)


class VTransformer(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: List[int],
        dropout_rate: float, act_fn: str="ELU", norm_type=None, lastlayer_reg: bool=True, resnet=True,
        attn_dropout_rate=0., attn_norm_type="LayerNorm",
    ):
        super(VTransformer, self).__init__()
        self.layers = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        input = input_size
        for i, hidden in enumerate(hidden_size):
            if i < len(hidden_size)-1 or lastlayer_reg:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
            else:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=0, act_fn=act_fn, norm_type=None))
            self.attn_layers.append(Attention2d(hidden, dropout_rate=attn_dropout_rate, norm_type=attn_norm_type))
            input = hidden
        self.output_layer = nn.Linear(input, 1)
        self.resnet = resnet
    def forward(self, x):
        for i in range(len(self.layers)):
            linear_out = self.layers[i](x)
            attn_out = self.attn_layers[i](linear_out)
            x = attn_out
            if self.resnet:
                x += linear_out
        return self.output_layer(x).squeeze()  # shape (N,)


class FeatureGroupTransformer(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: List[int],
        dropout_rate: float, act_fn: str="ELU", norm_type=None, lastlayer_reg: bool=True, resnet=True,
        n_group=4, attn_dropout_rate=0., attn_norm_type="LayerNorm",
    ):
        super(FeatureGroupTransformer, self).__init__()
        self.layers = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.n_group = n_group
        input = input_size
        for i, hidden in enumerate(hidden_size):
            if i < len(hidden_size)-1 or lastlayer_reg:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type))
            else:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=0, act_fn=act_fn, norm_type=None))
            assert hidden % n_group == 0, f"{hidden=} cannot be devided by {n_group=}"
            self.attn_layers.append(MultiHeadAttention(embed_dim=hidden//n_group, num_heads=1, dropout_rate=attn_dropout_rate, norm_type=attn_norm_type))
            input = hidden
        self.output_layer = nn.Linear(input, 1)
        self.resnet = resnet
    def forward(self, x):
        for i in range(len(self.layers)):
            linear_out = self.layers[i](x)  # (b, f)
            attn_in = linear_out.reshape(linear_out.shape[0], self.n_group, -1)  # (b, n_group, f//n_group)
            attn_out = self.attn_layers[i](attn_in)  # (b, n_group, f//n_group)
            attn_out_flatten = attn_out.reshape(attn_out.shape[0], -1)
            x = attn_out_flatten
            if self.resnet:
                x += linear_out
        return self.output_layer(x).squeeze()  # shape (N,)

class DenseBlockNEW(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0, act_fn=None, norm_type:str=None, require_linear=True, self_gating=False, gate_act_fn='Sigmoid', resnet=False):
        super(DenseBlockNEW, self).__init__()
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
        self.self_gating = self_gating
        if self_gating:
            self.gate = nn.Linear(input_size, output_size)
            self.gate_act_fn = getattr(nn, gate_act_fn)()
        self.resnet = resnet
        if resnet and input_size != output_size:
            self.resnet_proj = nn.Linear(input_size, output_size)
    def forward(self, x):
        if self.self_gating:
            out = self.block(x) * self.gate_act_fn(self.gate(x))
        else:
            out = self.block(x)
        if self.resnet:
            if hasattr(self, "resnet_proj"):
                x = self.resnet_proj(x)
            return x + out
        return out

class MLP3NEW(nn.Module):
    def __init__(self, input_size: int, hidden_size: List[int], dropout_rate: float, act_fn: str="ELU", norm_type=None, lastlayer_reg: bool=True, self_gating=False, gate_act_fn='Sigmoid', resnet=False, output_bias=True):
        super(MLP3NEW, self).__init__()
        self.layers = nn.ModuleList()
        input = input_size
        for i, hidden in enumerate(hidden_size):
            if i < len(hidden_size)-1 or lastlayer_reg:
                self.layers.append(DenseBlockNEW(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type, self_gating=self_gating, gate_act_fn=gate_act_fn, resnet=resnet))
            else:
                self.layers.append(DenseBlockNEW(input, hidden, dropout_rate=0, act_fn=act_fn, norm_type=None, self_gating=self_gating, gate_act_fn=gate_act_fn, resnet=resnet))
            input = hidden
        self.layers = nn.Sequential(*self.layers)
        self.output_layer = nn.Linear(input, 1, bias=output_bias)
    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x).squeeze()  # shape (N,)

class MLP3NEWFT(MLP3NEW):
    def __init__(self, *args, PT_model_path, free_copy_layers: List[int], additional_hidden_size: List[int], **kwargs,):
        super().__init__(*args, **kwargs)
        PT_model = torch.load(PT_model_path)
        for i, layer in enumerate(self.layers):
            PT_layer = PT_model.layers[i]
            for name, param in PT_layer.state_dict().items():
                layer.state_dict()[name].copy_(param)
            print(f"layer{i} param copied")
            if i in free_copy_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                print(f"layer{i} param frozen")
        if len(additional_hidden_size) > 0:
            self.additional_layers = nn.ModuleList()
            last_hidden_size = kwargs["hidden_size"][-1]
            input = last_hidden_size
            for i, hidden in enumerate(additional_hidden_size):
                self.additional_layers.append(DenseBlockNEW(input, hidden, dropout_rate=0, act_fn=kwargs["act_fn"], norm_type=None, self_gating=kwargs["self_gating"], gate_act_fn=kwargs["gate_act_fn"], resnet=kwargs["resnet"]))
                input = hidden
            self.additional_layers = nn.Sequential(*self.additional_layers)
            self.output_layer = nn.Linear(input, 1)  # 后面续了层就肯定要重新定义output_layer
        else:
            self.additional_layers = nn.Identity()
            for name, param in PT_model.output_layer.state_dict().items():
                self.output_layer.state_dict()[name].copy_(param)  # 继承output_layer的话就肯定不能freeze
    
    def forward(self, x):
        x = self.layers(x)
        x = self.additional_layers(x)
        return self.output_layer(x).squeeze()  # shape (N,)

class FusionGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float, num_layers: int,
                 group_sizes: List[int], fusion_type="concat", **kwargs):
        super(FusionGRU, self).__init__()
        assert sum(group_sizes) == input_size
        self.group_sizes = group_sizes
        self.fusion_type = fusion_type
        self.gru_models = nn.ModuleList()
        for group_size in group_sizes:
            self.gru_models.append(
                nn.GRU(
                    input_size=group_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout_rate,
                )
            )
        if fusion_type == "concat":
            gru_out_size = len(group_sizes) * hidden_size
            self.fc = nn.Sequential(
                nn.Linear(gru_out_size, gru_out_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(gru_out_size, 1)
            )
    def forward(self, x):
        # x (b, t, f)
        start_index = 0
        gru_outputs = []
        for i, group_size in enumerate(self.group_sizes):
            gru = self.gru_models[i]
            out, _ = gru(x[:, :, start_index: start_index+group_size])  # out (b, t, h)
            if self.fusion_type == "concat":
                gru_outputs.append(out[:, -1, :])
        gru_out = torch.cat(gru_outputs, axis=1)  # gru_out (b, h*n)
        out = self.fc(gru_out).squeeze()
        return out

class ThresholdSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold=0.5):
        """
        前向传播：执行硬选择操作
        """
        # 保存原始输入用于反向传播
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        
        # Sigmoid + 硬阈值选择
        sigmoid_x = torch.sigmoid(x)
        return (sigmoid_x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用 Sigmoid 的梯度作为近似值
        """
        x, = ctx.saved_tensors
        sigmoid_x = torch.sigmoid(x)
        grad_input = grad_output * sigmoid_x * (1 - sigmoid_x)  # 近似梯度
        return grad_input, None  # 第二个 None 是因为 threshold 不需要梯度

class ThresholdLayer(nn.Module):
    def __init__(self, threshold=0.5):
        super(ThresholdLayer, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        return ThresholdSTE.apply(x, self.threshold)

class TimingMLP(nn.Module):
    def __init__(self, input_size: int, softmax:bool, hard_select: bool, scaling:bool, hidden_size: List[int], dropout_rate: float, act_fn: str="ELU", norm_type=None, lastlayer_reg: bool=True):
        super(TimingMLP, self).__init__()
        self.layers = nn.ModuleList()
        input = input_size - 2
        for i, hidden in enumerate(hidden_size):
            if i < len(hidden_size)-1 or lastlayer_reg:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=dropout_rate, act_fn=act_fn, norm_type=norm_type,))
            else:
                self.layers.append(DenseBlock(input, hidden, dropout_rate=0, act_fn=act_fn, norm_type=None))
            input = hidden
        self.softmax = softmax
        if softmax:
            self.layers.extend([
                nn.Linear(input, 2),
                nn.Softmax(dim=-1)
            ])
        else:
            self.layers.append(nn.Linear(input, 1))
            if hard_select:
                self.weight_layer = ThresholdLayer(threshold=0.5)
            else:
                self.weight_layer = nn.Sigmoid()
        self.layers = nn.Sequential(*self.layers)
        
        self.scaling = scaling
        if scaling:
            self.signal1_scaling = nn.Linear(1, 1, bias=False)
            self.signal2_scaling = nn.Linear(1, 1, bias=False)
    
    def forward(self, x):
        features, signals = x[:, :-2], x[:, -2:]
        if self.scaling:
            signal1 = self.signal1_scaling(signals[:, 0:1])
            signal2 = self.signal2_scaling(signals[:, 1:2])
        else:
            signal1 = signals[:, 0:1]
            signal2 = signals[:, 1:2]
        if self.softmax:
            softmax_out = self.layers(features)  # (B, 2)
            signals = torch.cat([signal1, signal2], dim=1)  # (B, 2)
            output = (softmax_out * signals).sum(axis=1)  # (B,)
        else:
            logits = self.layers(features).squeeze(-1)  # (B,)
            weight = self.weight_layer(logits)  # (B,)
            output = weight * signal1.squeeze(-1) + (1 - weight) * signal2.squeeze(-1)  # (B,)
        return output



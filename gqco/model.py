import math
from ml_collections import ConfigDict
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, GELU
from torch.nn import functional as F
from torch.nn.modules import *
from torch.nn.modules.transformer import _get_activation_fn
from typing import Any, Callable, Optional, Union  ## <- not used explicitly but needed for decoder layer class

from torch_geometric.nn import GINEConv, TransformerConv, ResGatedGraphConv, GATv2Conv, CGConv, PDNConv, SplineConv, NNConv, PNAConv

from gqco.utils import print0
from gqco.data import compute_node_dim, compute_edge_dim








def define_model(args):

    model = TransformerWithMoE(args)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params_en = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    pytorch_total_params_de = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

    ## if expert tuning
    if args.tune_size > 0:
        model = grad_true_for_expert_tuning(model, args.tune_size, args)

    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_trainable_params_en = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    pytorch_trainable_params_de = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)


    print0(f'------------------------ Training info ------------------------', args)
    print0(f'--- total params: {pytorch_total_params / 1e6:.2f}M', args)
    print0(f'--- total params (encoder): {pytorch_total_params_en / 1e6:.2f}M', args)
    print0(f'--- total params (decoder): {pytorch_total_params_de / 1e6:.2f}M', args)
    print0(f'--- total trainable params: {pytorch_trainable_params / 1e6:.2f}M', args)
    print0(f'--- total trainable params (encoder): {pytorch_trainable_params_en / 1e6:.2f}M', args)
    print0(f'--- total trainable params (decoder): {pytorch_trainable_params_de / 1e6:.2f}M', args)
    print0(f'--- vacab size: {args.vocab_size}', args)
    print0(f'--- World size: {args.world_size}', args)
    # print0(f'--- Device: {args.device}', args)
    print0(f'-------------------- end of Training info ---------------------', args)
    print0('\n', args)

    return model, args






class ConvLayer(torch.nn.Module):
    def __init__(self, conv_type, hidden_dim, heads=1, bias=False, dropout=0.0, max_size=20):
        super(ConvLayer, self).__init__()

        self.conv_type = conv_type

        if conv_type == 'GINE':
            self.conv = GINEConv(
                    Sequential(
                        Linear(hidden_dim, hidden_dim*4, bias=bias),
                        GELU(),
                        Linear(hidden_dim*4, hidden_dim, bias=bias)
                    ),
                    edge_dim=hidden_dim
                )

        else:
            if self.conv_type == 'Transformer':
                self.conv = TransformerConv(hidden_dim, hidden_dim // heads, edge_dim=hidden_dim, heads=heads, bias=bias)
            if self.conv_type == 'ResGatedGraph':
                self.conv = ResGatedGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim, bias=bias)
            if self.conv_type == 'GATv2':
                self.conv = GATv2Conv(hidden_dim, hidden_dim // heads, edge_dim=hidden_dim, heads=heads, bias=bias)
            if self.conv_type == 'CG':
                self.conv = CGConv(hidden_dim, dim=hidden_dim, bias=bias)
            if self.conv_type == 'PDN':
                self.conv = PDNConv(hidden_dim, hidden_dim, edge_dim=hidden_dim, hidden_channels=hidden_dim*4, bias=bias)
            if conv_type == 'Spline':
                self.conv = SplineConv(hidden_dim, hidden_dim, kernel_size=2, dim=hidden_dim, bias=bias)
            if conv_type == 'NN':
                self.conv = NNConv(
                    hidden_dim,
                    hidden_dim,
                    Sequential(
                        Linear(hidden_dim, hidden_dim*4, bias=bias),
                        GELU(),
                        Linear(hidden_dim*4, hidden_dim**2, bias=bias)
                    ),
                    bias=bias
                )
            if conv_type == 'PNA':
                self.conv = PNAConv(
                    hidden_dim, 
                    hidden_dim, 
                    aggregators=['sum', 'mean', 'min', 'max', 'var', 'std'], 
                    scalers=['identity', 'amplification', 'attenuation', 'linear', 'inverse_linear'], 
                    deg = torch.tensor([0 for _ in range(3)]+[1]),
                    edge_dim=hidden_dim,
                    act='gelu'
                )

        self.ffns = nn.ModuleDict({
            f'Expert-{size}': FFN_Expert(hidden_dim, hidden_dim*4) for size in range(3, max_size+1)
        })
        self.norm1 = LayerNorm(hidden_dim, eps=1e-5, bias=True)
        self.norm2 = LayerNorm(hidden_dim, eps=1e-5, bias=True)

        self.dr1 = nn.Dropout(dropout)
        self.dr2 = nn.Dropout(dropout)


    def forward(self, x=None, edge_index=None, edge_attr=None, batch=None, record=None, size=None):

        if self.conv_type == 'GINE':
            x = x + self.dr1(self.conv(x, edge_index, edge_attr))
            x = self.norm1(x)

        else:
            x = self.dr1(self.conv(x, edge_index, edge_attr))
            x = self.norm1(x)
            x = x + self.dr2(self.ffns[f'Expert-{size}'](x))
            x = self.norm2(x)

        return x



class ConvNetwork(torch.nn.Module):
    def __init__(self, conv_type, hidden_dim, num_layers, heads=1, bias=False, dropout=0.0, max_size=20):
        super(ConvNetwork, self).__init__()

        self.convs = nn.ModuleList([
            ConvLayer(conv_type, hidden_dim, heads, bias, dropout, max_size) for _ in range(num_layers)
        ])


    def forward(self, x=None, edge_index=None, edge_attr=None, batch=None, record=None, size=None):

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr, batch, record, size)

        return x







class MyEncoder(nn.Module):

    def __init__(self, configs):

        super().__init__()

        self.max_size = configs.max_size
        self.hidden_dim = configs.hidden_dim

        self.encoder_type = configs.encoder_type
        self.encoder_hidden_dim = configs.encoder_hidden_dim
        self.dropout = configs.dropout


        self.ffns_in = nn.ModuleDict({
            f'Expert-{s}': 
            FFN_Expert(
                d_model = compute_node_dim(size=s), 
                inner_dim = configs.hidden_dim*4, 
                d_out = configs.encoder_hidden_dim, 
                bias = False, 
                layer_norm = False, 
                dropout = configs.dropout
            ) for s in range(3, configs.max_size+1)
        })
        self.ffns_in_e = nn.ModuleDict({
            f'Expert-{s}': 
            FFN_Expert(
                d_model = compute_edge_dim(size=s), 
                inner_dim = configs.hidden_dim*4, 
                d_out = configs.encoder_hidden_dim, 
                bias = False,
                layer_norm = False, 
                dropout = configs.dropout
            ) for s in range(3, configs.max_size+1)
        })

        self.dropout_in = nn.Dropout(self.dropout)
        self.dropout_in_e = nn.Dropout(self.dropout)
        self.norm_in = LayerNorm(configs.encoder_hidden_dim, eps=1e-5, bias=True)
        self.norm_in_e = LayerNorm(configs.encoder_hidden_dim, eps=1e-5, bias=True)

        self.convolution = ConvNetwork(
            conv_type = configs.encoder_type,
            hidden_dim = configs.encoder_hidden_dim, 
            num_layers = configs.depth,
            heads = configs.num_heads,
            dropout = configs.dropout,
            bias = False,
            max_size = configs.max_size
        )    



    def forward(self, record):

        size = record['size']
        num_clone = record['len']

        ## Input embedding
        x = self.ffns_in[f'Expert-{size}'](record.x)  ## (size*num_clone, node_feat) -> (size*num_clone, en_hidden_dim)
        e = self.ffns_in_e[f'Expert-{size}'](record.edge_attr)  ## (size*num_clone, edge_feat) -> (size*num_clone, en_hidden_dim)
        x = self.dropout_in(x)
        e = self.dropout_in_e(e)
        x = self.norm_in(x)
        e = self.norm_in_e(e)

        ## Graph encoding
        h = self.convolution(x=x, edge_index=record.edge_index, edge_attr=e, batch=record.batch, size=size)  ## (size*num_clone, en_hidden_dim) -> (size*num_clone, en_hidden_dim)
        h = torch.reshape(h, (num_clone, size, self.encoder_hidden_dim)) ## (size*num_clone, en_hidden_dim) -> (num_clone, size, en_hidden_dim)


        return h






class FFN_Expert(nn.Module):

    def __init__(self, d_model, inner_dim, d_out=None, dropout=0.0, bias=False, layer_norm=False):
        super(FFN_Expert, self).__init__()

        self.layer_norm = layer_norm

        self.ffn1 = nn.Linear(d_model, inner_dim, bias=bias)

        if d_out is None:
            self.ffn2 = nn.Linear(inner_dim, d_model, bias=bias)
        else:
            self.ffn2 = nn.Linear(inner_dim, d_out, bias=bias)

        self.dropout = nn.Dropout(dropout)

        if layer_norm:
            self.norm = LayerNorm(d_out, eps=1e-5, bias=True)


    def forward(self, x):

        x = self.ffn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.ffn2(x)

        if self.layer_norm:
            x = self.norm(x)
        
        return x



    



class PositionalEmbedding(nn.Module):

    def __init__(self, max_seq_len, embed_model_dim):
        super(PositionalEmbedding, self).__init__()

        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
      
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)

        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)

        return x
    





## Copied from pytorch implementation
## Ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py
## Customized for k-qubit-MoE
class TransformerDecoderLayer(nn.Module):

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        max_size,
        dim_feedforward = 2048,
        dropout = 0.1,
        activation = F.gelu,
        layer_norm_eps = 1e-5,
        batch_first = False,
        norm_first = False,
        bias = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        # self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        # self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        # self.ffns = nn.ModuleList([
        #     FFN_Expert(d_model, dim_feedforward) for _ in range(max_size+1)
        # ])
        self.ffns = nn.ModuleDict({
            f'Expert-{s}': 
            FFN_Expert(d_model, dim_feedforward) for s in range(3, max_size+1)
        })


        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask = None,
        memory_mask = None,
        tgt_key_padding_mask = None,
        memory_key_padding_mask = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ):

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        size = memory.shape[1]

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x, size))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x, size))

        return x

    # self-attention block
    def _sa_block(
        self,
        x,
        attn_mask,
        key_padding_mask,
        is_causal: bool = False,
    ):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x,
        mem,
        attn_mask,
        key_padding_mask,
        is_causal: bool = False,
    ):
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x, size):
        # x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # return self.dropout3(x)
        # x = self.ffns[size](x)
        x = self.ffns[f'Expert-{size}'](x)
        return self.dropout3(x)








class TransformerWithMoE(nn.Module):

    def __init__(self, args):
        super(TransformerWithMoE, self).__init__()

        self.max_generation = args.max_generation
        self.min_generation = args.min_generation
        self.vocab_size = args.vocab_size

        ## Encoder
        encoder_config = ConfigDict()
        encoder_config.encoder_type = args.encoder_type
        encoder_config.max_size = args.max_size
        encoder_config.encoder_hidden_dim = args.encoder_hidden_dim_inner
        encoder_config.hidden_dim = args.hidden_dim
        encoder_config.depth = args.encoder_depth
        encoder_config.num_heads = args.encoder_num_heads
        encoder_config.dropout = 0.0
        self.encoder_config = encoder_config


        ## Decoder
        decoder_config = ConfigDict()
        decoder_config.d_model = args.hidden_dim
        decoder_config.n_head = args.decoder_num_heads
        decoder_config.vocab_size = args.vocab_size
        decoder_config.pad_token_id = 0
        decoder_config.eos_token_id = args.eos_token_id
        decoder_config.decoder_start_token_id = 0
        decoder_config.decoder_layers = args.decoder_depth
        decoder_config.decoder_ffn_dim = args.decoder_ffn_dim
        decoder_config.dropout = 0.0
        self.decoder_config = decoder_config


        ## Model definition
        self.encoder = MyEncoder(self.encoder_config)
        self.embedding = nn.Embedding(args.vocab_size, args.hidden_dim)
        self.position_embedding = PositionalEmbedding(20, args.hidden_dim)
        decoder_layer = TransformerDecoderLayer(
            d_model = args.hidden_dim,
            nhead = args.decoder_num_heads,
            max_size = args.max_size,
            dim_feedforward = args.decoder_ffn_dim,
            dropout = 0.0,
            batch_first = True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.decoder_depth)
        self.lms = nn.ModuleDict({
            f'Expert-{s}':
            nn.Linear(args.hidden_dim, args.vocab_size, bias=False) for s in range(3, args.max_size+1)
        })



    def step(self, record, len_record, decoder_input_ids=None):

        ## Encoder forward pass
        encoder_outputs = self.encoder(record)

        if decoder_input_ids is None:
            decoder_input_ids = torch.full((len_record, 1), self.decoder_config.decoder_start_token_id)

        ## Decoder forward pass
        decoder_inputs = self.position_embedding(self.embedding(decoder_input_ids))
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs)

        ## Linear head
        logits = self.lms[f'Expert-{record["size"]}'](decoder_outputs)

        return logits
    



    def forward(self, record, temperature=1, masked_tokens=[], same_token_penalty=0):

        self.device = record['x'].device
        len_record = record['len']
        max_generation = int(record['size']*2)


        tokens = torch.full((len_record, 1), self.decoder_config.decoder_start_token_id).to(self.device)
        probs_all = torch.full((len_record, 1, self.decoder_config.vocab_size), 0, device=self.device)
        logits_all = torch.full((len_record, 1, self.decoder_config.vocab_size), 0, device=self.device)
        probs_token = torch.full((len_record, 1), 0, device=self.device)
        finished_sequences = torch.zeros(len_record, dtype=torch.bool, device=self.device)

        ## Generation
        for _len in range(max_generation):
            
            ## Forward
            out_logits = self.step(record, len_record, decoder_input_ids=tokens)

            ## Mask unused tokens
            masked_logits = out_logits[:, -1, :]
            if _len < self.min_generation:   ## Do not generate eos token until min_generation
                masked_logits[:, torch.tensor([0])] = float('-inf')
            if len(masked_tokens) > 0:   ## Do not generate bad words
                masked_logits[:, torch.tensor(masked_tokens)] = float('-inf')
            if _len > 0:  ## Do not to generate the same token as just one before
                _mask = torch.zeros_like(masked_logits.detach())
                _mask.scatter_(1, next_tokens, float(same_token_penalty))
                masked_logits += _mask

            ## Token generation
            probs = F.softmax(masked_logits/temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            ## Store the probabilities of the selected tokens
            selected_probs = probs.gather(1, next_tokens)
            selected_probs = torch.where(finished_sequences.unsqueeze(1), torch.tensor(1.0, device=self.device), selected_probs.clone())
            probs_token = torch.cat([probs_token, selected_probs], dim=1)

            ## Replace token with eos_token_id for finished seq
            next_tokens = torch.where(finished_sequences.unsqueeze(1), self.decoder_config.eos_token_id, next_tokens)

            ## Concatenate
            tokens = torch.cat([tokens, next_tokens], dim=1)
            probs_all = torch.cat([probs_all, probs.unsqueeze(1)], dim=1)
            logits_all = torch.cat([logits_all, masked_logits.unsqueeze(1)], dim=1)

            ## Check finish
            if (_len >= self.min_generation):
                finished_sequences |= next_tokens.squeeze(1) == 0

            if finished_sequences.all():
                break

        ## Add eos token for max generated seq
        if _len == max_generation-1:
            tokens = torch.cat([tokens, torch.zeros_like(next_tokens)], dim=1)

        return tokens[:, 1:], probs_all[:, 1:], probs_token[:, 1:], logits_all[:, 1:]






def grad_true_for_expert_tuning(model, size, args):

    for p in model.parameters():
        p.requires_grad = False

    ## Encoder
    for p in model.encoder.ffns_in_e[f'Expert-{size}'].parameters():
        p.requires_grad = True
    for p in model.encoder.ffns_in[f'Expert-{size}'].parameters():
        p.requires_grad = True

    for j in range(len(model.encoder.convolution.convs)):
        for p in model.encoder.convolution.convs[j].ffns[f'Expert-{size}'].parameters():
            p.requires_grad = True

    ## Decoder
    for j in range(len(model.decoder.layers)):
        for p in model.decoder.layers[j].ffns[f'Expert-{size}'].parameters():
            p.requires_grad = True

    ## Outlayer
    model.lms[f'Expert-{size}'].weight.requires_grad = True

    return model



def grad_false_for_unused_experts(model, size, args):

    ## reset requires_grad
    for p in model.parameters():
        p.requires_grad = True


    ## set false for unused expert parameters
    for s in range(3, args.max_size+1):
        if s != size:

            ## Encoder
            for p in model.encoder.ffns_in_e[f'Expert-{s}'].parameters():
                p.requires_grad = False
            for p in model.encoder.ffns_in[f'Expert-{s}'].parameters():
                p.requires_grad = False

            for j in range(len(model.encoder.convolution.convs)):
                for p in model.encoder.convolution.convs[j].ffns[f'Expert-{s}'].parameters():
                    p.requires_grad = False

            ## Decoder
            for j in range(len(model.decoder.layers)):
                for p in model.decoder.layers[j].ffns[f'Expert-{s}'].parameters():
                    p.requires_grad = False

            ## Outlayer
            model.lms[f'Expert-{s}'].weight.requires_grad = False

    return model






def param_clone(model, old_size, new_size, device='cpu'):

    with torch.no_grad():

        ## Encoder
        copy_params_with_filling(model.encoder.ffns_in[f'Expert-{old_size}'].ffn1.weight,   model.encoder.ffns_in[f'Expert-{new_size}'].ffn1.weight, device)
        copy_params_with_filling(model.encoder.ffns_in[f'Expert-{old_size}'].ffn2.weight, model.encoder.ffns_in[f'Expert-{new_size}'].ffn2.weight, device)
        copy_params_with_filling(model.encoder.ffns_in_e[f'Expert-{old_size}'].ffn1.weight,   model.encoder.ffns_in_e[f'Expert-{new_size}'].ffn1.weight, device)
        copy_params_with_filling(model.encoder.ffns_in_e[f'Expert-{old_size}'].ffn2.weight, model.encoder.ffns_in_e[f'Expert-{new_size}'].ffn2.weight, device) 

        for j in range(len(model.encoder.convolution.convs)):
            model.encoder.convolution.convs[j].ffns[f'Expert-{new_size}'].load_state_dict(model.encoder.convolution.convs[j].ffns[f'Expert-{old_size}'].state_dict())
        

        ## Decoder
        for j in range(len(model.decoder.layers)):
            model.decoder.layers[j].ffns[f'Expert-{new_size}'].load_state_dict(model.decoder.layers[j].ffns[f'Expert-{old_size}'].state_dict())
    
        
        ## Outlayer
        model.lms[f'Expert-{new_size}'].weight.copy_(model.lms[f'Expert-{old_size}'].weight)

        torch.cuda.empty_cache()

    return model





def copy_params_with_filling(param1, param2, device):
    # Initialize on CPU to save GPU memory
    if param1.shape[1] < param2.shape[1]:
        mean, std = param1.detach().mean().item(), param1.detach().std().item()
        new_param = torch.normal(mean, std, size=param2.size()).cpu()
        new_param[:, :param1.size(1)] = param1.detach().cpu()
        param2.copy_(new_param.to(device))  # Move to GPU after filling
    param2.requires_grad = True
    torch.cuda.empty_cache()

from typing import Tuple
import torch
from torch import nn
from torch.nn.utils import spectral_norm as spec_norm
from torch.nn import functional as F
import numpy as np

from ...setup_logger import get_logger
logger = get_logger(__name__)

from ..positional_embedding import TimestepEmbedding

def get_linear(n_in, n_out):
    layer = nn.Linear(n_in, n_out)
    nn.init.xavier_uniform(layer.weight.data, 1.0)
    return layer


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        n_out: int,
        n_classes: int,
        n_hidden: int, 
        n_layers: int = 4,
        pos_embedding: bool = True,
        norm_layer: str = "layer_norm",
        spec_norm: bool = False,
    ):
        super().__init__()
        if spec_norm:
            logger.info("G: spectral norm enabled")
            sn_fn = spec_norm
        else:
            sn_fn = lambda x: x

        if norm_layer == "layer_norm":
            layer_norm_fn = nn.LayerNorm
        elif norm_layer == "batch_norm":
            layer_norm_fn = nn.BatchNorm1d
        elif norm_layer is None:
            layer_norm_fn = nn.Identity
        else:
            raise NotImplementedError("{} unknown".format(norm_layer))

        encoder = []
        for j in range(n_layers+1):
            if j == 0:
                this_in, this_out = n_out+n_hidden+n_out, n_hidden
            elif j == (n_layers):
                this_in, this_out = n_hidden*2+n_out, n_hidden
            else:
                this_in, this_out = n_hidden*2+n_out, n_hidden
            encoder.append(nn.Sequential(
                sn_fn(nn.Linear(this_in, this_out)),
                layer_norm_fn(this_out),
                #nn.BatchNorm1d(this_out) if use_norm else nn.Identity(),
                nn.ReLU() if j != n_layers else nn.Identity()
            ))
        self.encoder = nn.ModuleList(encoder)

        self.n_out = n_out

        if pos_embedding:
            self.y_embed = TimestepEmbedding(
                embedding_dim=n_hidden,
                hidden_dim=n_hidden,
                output_dim=n_out
            )
            self.t_embed = TimestepEmbedding(
                embedding_dim=n_hidden,
                hidden_dim=n_hidden,
                output_dim=n_hidden
            )            
        else:
            self.y_embed = nn.Linear(1, n_hidden)
            self.t_embed = nn.Linear(1, n_hidden)

        decoder = []
        for j in range(n_layers+1):
            if j == 0:
                this_in, this_out = n_hidden*2+n_out, n_hidden
            elif j == (n_layers):
                this_in, this_out = n_hidden*2+n_out, n_out
            else:
                this_in, this_out = n_hidden*2+n_out, n_hidden
            decoder.append(nn.Sequential(
                sn_fn(nn.Linear(this_in, this_out)),
                #nn.BatchNorm1d(this_out) if use_norm else nn.Identity(),
                layer_norm_fn(this_out),
                nn.ReLU() if j != n_layers else nn.Identity()
            ))
        self.decoder = nn.ModuleList(decoder)

    def embed(self, y):
        """Embed y to be in the same dimension as x"""
        return self.y_embed(y)

    def encode(self, x, y_emb, t):
        if type(y_emb) == torch.LongTensor:
            raise Exception("encode() takes y already pre-embedded, use self.embed_y")
        h = x
        t_emb = self.t_embed(t)
        for b,mod in enumerate(self.encoder):
            #print(b, "-->", h.shape, t_emb.shape, y_emb.shape)
            h = mod( torch.cat((h, t_emb, y_emb), dim=1) )
        return h

    def decode(self, z, y_emb, t):
        h = z
        t_emb = self.t_embed(t)
        for mod in self.decoder:
            h = mod( torch.cat((h, t_emb, y_emb), dim=1) )
        return h

    def forward(self, x, y_emb, t):
        this_z = self.encode(x, y_emb, t)
        return self.decode(this_z, y_emb, t)

    @torch.no_grad()
    def sample(self, y_cond, z=None):
        self.eval()
        raise Exception("TODO")

if __name__ == '__main__':
    EncoderDecoder(z_dim=32, n_hidden=256, n_out=16, discrete=False)
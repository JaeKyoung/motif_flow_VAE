import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from motifflow.models.latent.utils import get_positional_embedding
from timm.models.vision_transformer import Attention, Mlp
from typing import Dict, Optional

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

########################################################
# following code is modified from https://github.com/facebookresearch/DiT/blob/main/models.py
########################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# TODO : RoPE (Rotary Position Embedding)


#########################################################
# Core DiT Model
#########################################################

class ProteinLatentDiTBlock(nn.Module):
    """
    A Protein Latent DiT block with adaptive layer norm zero (adaLN-Zero) conditioning
    following the original DiT.
    """
    def __init__(
            self,
            hidden_size = 512,
            num_heads = 16,
            mlp_ratio = 4.0,
            dropout = 0.0,
            residual_dropout = 0.0,
            **block_kwargs
        ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True,
                              attn_drop=dropout, proj_drop=dropout, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, z_emb, cond):
        # Conditioning (B, hidden_size)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa = shift_msa.squeeze(1), scale_msa.squeeze(1), gate_msa.squeeze(1)
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.squeeze(1), scale_mlp.squeeze(1), gate_mlp.squeeze(1)
        # self-attention
        z_emb = z_emb + gate_msa.unsqueeze(1) * self.residual_dropout(self.attn(modulate(self.norm1(z_emb), shift_msa, scale_msa)))
        # MLP
        z_emb = z_emb + gate_mlp.unsqueeze(1) * self.residual_dropout(self.mlp(modulate(self.norm2(z_emb), shift_mlp, scale_mlp)))
        return z_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, final_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, final_dim, bias=True)

    def forward(self, z_emb, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        shift, scale = shift.squeeze(1), scale.squeeze(1)
        z_emb = modulate(self.norm_final(z_emb), shift, scale)
        z_emb = self.linear(z_emb)
        return z_emb
    

class ProteinLatentDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone working on latent vectors.

    Inputs:
        z: (B, N, latent_dim) tensor of latent vectors
        t: (B,) tensor of diffusion timesteps
        motif_mask: (B, N,) tensor of motif mask
        
    Outputs:
        z: (B, N, latent_dim) tensor of latent vectors generated by the model

    Configuration (cfg):
        - latent_dim
        - hidden_size
        - depth
        - num_heads
        - mlp_ratio
        - dropout
        - learn_sigma
    """
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.hidden_size = cfg.hidden_size
        self.depth = cfg.depth
        self.num_heads = cfg.num_heads
        self.mlp_ratio = cfg.mlp_ratio
        self.learn_sigma = cfg.learn_sigma
        self.dropout = cfg.dropout
        self.residual_dropout = cfg.residual_dropout
        self.z_embedder = nn.Linear(self.latent_dim, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)

        self.blocks = nn.ModuleList([
            ProteinLatentDiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio, self.dropout, self.residual_dropout)
            for _ in range(self.depth)
        ])

        # If learn_sigma is True, final layer outputs 2 * latent_dim; otherwise latent_dim.
        final_out_dim = self.latent_dim * 2 if self.learn_sigma else self.latent_dim
        self.final_layer = FinalLayer(self.hidden_size, final_out_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # TODO...
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)
        return ckpt_forward

    def forward(
        self, 
        z: torch.Tensor,
        t: torch.Tensor,
        motif_mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of protein latent DiT
        """
        B, num_res, _ = z.shape

        # Project latent vectors and add positional embedding.
        z_emb = self.z_embedder(z) # (B, N, hidden_size)
        residue_indices = torch.arange(num_res, device=z.device)
        pos_embed = get_positional_embedding(residue_indices, self.hidden_size, max_len=2056)
        pos_embed = pos_embed * motif_mask.unsqueeze(-1)
        z_emb = z_emb + pos_embed
        
        # Timestep embedding conditioning + add other conditionings in here such as label (y)
        t_emb = self.t_embedder(t) # (B, hidden_size)
        # y = self.y_embedder(y) # for label-conditioned generation
        cond = t_emb # + y
        cond = cond.unsqueeze(1) # (B, 1, hidden_size) apply same timestep embedding to all residues
        
        # Apply blocks
        for block in self.blocks:
            z_emb = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), z_emb, cond, use_reentrant=False)
        
        # Final projection to latent space
        # If learn_sigma is True, final_layer outputs 2 * latent_dim.
        z_out = self.final_layer(z_emb, cond) # (B, N, final_out_dim)
        
        return z_out
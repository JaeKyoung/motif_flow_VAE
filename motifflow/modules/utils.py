import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from openfold.utils.rigid_utils import Rigid

def calc_distogram(pos, min_bin, max_bin, num_bins):
    """
    Calculate a distogram from 3D positions.

    Args:
        pos: 3D positions of shape [B, N, 3]
        min_bin: Minimum distance for the first bin
        max_bin: Maximum distance for the last bin
        num_bins: Number of bins in the distogram

    Returns:
        Distogram of shape [B, N, N, num_bins]
    """
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram

def get_index_embedding(indices, embed_size, max_len=2056):
    """
    Create sine/cosine positional embeddings from prespecified indices.

    Args:
        indices: Offsets of size [..., N_edges] of type integer
        embed_size: Dimension of the embeddings to create
        max_len: Maximum length for the embedding calculation

    Returns:
        Positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding

def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    """
    Create time embeddings for diffusion models.

    Args:
        timesteps: Tensor of timesteps
        embedding_dim: Dimension of the time embedding
        max_position: Maximum number of positions for the embedding calculation

    Returns:
        Time embedding of shape [len(timesteps), embedding_dim]
    """
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    
    assert len(timesteps.shape) == 1, f"Expected 1D tensor, got shape {timesteps.shape}"

    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def distance(p, eps=1e-10):
    # from genie.utils.geo_utils
    """
    Compute distances between pairs of Euclidean coordinates.

    Args:
        p:
            [*, 2, 3] Input tensor where the last two dimensions have 
            a shape of [2, 3], representing a pair of coordinates in 
            the Euclidean space.

    Returns:
        [*] Output tensor of distances, where each distance is computed
        between the pair of Euclidean coordinates in the last two 
        dimensions of the input tensor p.
    """
    return (eps + torch.sum((p[..., 0, :] - p[..., 1, :]) ** 2, dim=-1)) ** 0.5

def rot_to_quat(rot: torch.Tensor,):
    # from genie.utils.affine_utils
    if(rot.shape[-2:] != (3, 3)):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot 

    k = [
        [ xx + yy + zz,      zy - yz,      xz - zx,      yx - xy,],
        [      zy - yz, xx - yy - zz,      xy + yx,      xz + zx,],
        [      xz - zx,      xy + yx, yy - xx - zz,      yz + zy,],
        [      yx - xy,      xz + zx,      yz + zy, zz - xx - yy,]
    ]

    k = (1./3.) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)

    _, vectors = torch.linalg.eigh(k)
    return vectors[..., -1]


def cal_distogram(
    positions: torch.Tensor,
    min_bin: float = 1e-3,
    max_bin: float = 20.0,
    num_bins: int = 22,
) -> torch.Tensor:
    """Calculate frame distogram.

    Parameters
    ----------
    positions: FloatTensor, [B, L, 3]
        Tensor of each frame positions.
    min_bin: float, default = 1e-3
        Minimum distance for get bin.
    max_bin: flaot, default = 20.0
        Maximum distance for get bin.
    num_bins: int, default = 22
        Number of bins for distogram.

    Returns
    -------
    dgram: LongTensor, [B, L, L, num_binds]
        Distogram for frame positions.
    """
    device = positions.device

    dists_2d = torch.linalg.norm(
        positions.unsqueeze(-2) - positions.unsqueeze(-3), axis=-1
    )[..., None]
    lower = torch.linspace(min_bin, max_bin, num_bins).to(device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).long()
    return dgram

def cal_unit_vector(rigids: Rigid, eps=1e-20):
    points = rigids.get_trans()[..., None, :, :]
    rigid_vec = rigids[..., None].invert_apply(points)

    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec**2, dim=-1))
    unit_vector = rigid_vec * inv_distance_scalar[..., None]
    unit_vector = torch.unbind(unit_vector[..., None, :], dim=-1)
    unit_vector = torch.cat(unit_vector, dim=-1)
    return unit_vector
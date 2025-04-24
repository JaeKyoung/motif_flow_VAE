import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from importlib.util import find_spec
from scipy.stats import truncnorm
from typing import Optional, Callable

is_deepspeed_available = False # find_spec("deepspeed") is not None
if is_deepspeed_available:
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention


def sigmoid_gate(gate, rep):
    return nn.Sigmoid()(gate) * rep


def swish(x, y):
    return x * torch.sigmoid(x) * y


class RelativePositionEmbedding(nn.Module):
    """Relative position embedding module.

    Relative position embedding is used to encode the relative position of residues.
    This module is came from ESM3.

    Parameters
    ----------
    d_hidden: int
        Dimension of hidden layer.
    bins: int
        Number of bins for relative position encoding.
    init_std: float
        Standard deviation for embedding initialization.
    """

    def __init__(self, d_hidden: int, bins: int = 32, init_std: float = 0.02):
        super().__init__()
        self.bins = bins

        self.embedding = nn.Embedding(2 * bins + 2, d_hidden)
        self.embedding.weight.data.normal_(0, init_std)

    def forward(
        self,
        seq_idx: torch.LongTensor,  # (..., L)
    ) -> torch.Tensor:  # (..., L, L, d_model)
        diff = seq_idx.unsqueeze(-1) - seq_idx.unsqueeze(-2)
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # add 1 to adjust for padding index
        output = self.embedding(diff)
        return output


class AttentionPairBias(nn.Module):
    """Attention with pair bias (Algorithm 24).

    Parameters
    ----------
    d_single: int
        Dimension of single representation.
    d_pair: int
        Dimension of pair representation.
    n_head: int
        Number of attention heads.
    use_deepspeed: bool
        Whether to use DeepSpeed EvoformerAttention.
    """

    def __init__(
        self,
        d_single: int = 256,
        d_pair: int = 128,
        n_head: int = 8,
        use_deepspeed: bool = False,
    ):
        super().__init__()
        self.n_head = n_head
        self.use_deepspeed = False  # use_deepspeed
        if use_deepspeed and not is_deepspeed_available:
            raise ValueError("DeepSpeed is not available.")

        assert d_single % n_head == 0, f"{d_single=} must be divisible by {n_head=}"
        d_hidden = d_single // n_head
        self.scaling = 1 / math.sqrt(d_hidden)

        self.ln_single = nn.LayerNorm(d_single)
        self.to_query = Linear(d_single, d_hidden * n_head, init="glorot")
        self.to_key = Linear(d_single, d_hidden * n_head, False, init="glorot")
        self.to_value = Linear(d_single, d_hidden * n_head, False, init="glorot")

        self.ln_pair = nn.LayerNorm(d_pair)
        self.to_bias = Linear(d_pair, n_head, False, init="default")
        self.to_gate = Linear(d_single, n_head, False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_single, False, init="final")

    def forward(
        self,
        single: torch.Tensor,  # (B, L, d_single)
        pair: torch.Tensor,  # (B, L, L, d_pair)
        mask: Optional[torch.Tensor] = None,  # (B, L)
    ) -> torch.Tensor:  # (B, L, d_single)
        B, L, _ = single.shape

        # Compute QKV
        single = self.ln_single(single)
        query = self.to_query(single)
        key = self.to_key(single)
        value = self.to_value(single)

        # Get bias using pair
        pair = self.ln_pair(pair)
        bias = self.to_bias(pair)

        if self.use_deepspeed:
            # Split heads
            query = query.view(B, 1, L, self.n_head, -1)
            key = key.view(B, 1, L, self.n_head, -1)
            value = value.view(B, 1, L, self.n_head, -1)

            # Convert to half precision for DeepSpeed compatibility
            query = query.half()
            key = key.half()
            value = value.half()
            bias = bias.half().permute(0, 3, 1, 2).unsqueeze(1)
            if mask is not None:
                mask = mask.half().unsqueeze(1).unsqueeze(1).unsqueeze(1)

            # Compute attention
            output = DS4Sci_EvoformerAttention(query, key, value, [mask, bias])
            output = output.float().squeeze(1)
        else:
            # Split heads
            query = query.view(B, L, self.n_head, -1)
            key = key.view(B, L, self.n_head, -1)
            value = value.view(B, L, self.n_head, -1)

            # Compute attention scores
            query.mul_(self.scaling)
            attention = torch.einsum("blhd,bkhd->bhlk", query, key)
            attention = attention + bias.permute(0, 3, 1, 2)

            # Apply the mask
            if mask is not None:
                attention.masked_fill_(~mask[:, None, None], float("-inf"))

            # Compute attention
            attention_weights = torch.softmax(attention, dim=-1)
            output = torch.einsum("bhlk,bkhd->blhd", attention_weights, value)

        # Apply the gate & project back to the original dimension
        gate = self.to_gate(single).unsqueeze(-1)
        output = sigmoid_gate(gate, output)
        output = output.view(B, L, -1)
        output = self.to_out(output)
        return output


class Transition(nn.Module):
    """Transition layer (Algorithm 11).

    Parameters
    ----------
    d_hidden: int
        Dimension of the input and output features.
    n: int
        Expansion factor.
    """

    def __init__(self, d_hidden: int = 128, n: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(d_hidden)
        self.expand_a = Linear(d_hidden, d_hidden * n, False, init="default")
        self.expand_b = Linear(d_hidden, d_hidden * n, False, init="default")
        self.squeeze = Linear(d_hidden * n, d_hidden, False, init="final")

    def forward(
        self,
        x: torch.Tensor,  # (..., d_hidden)
    ) -> torch.Tensor:  # (..., d_hidden)
        x = self.ln(x)
        a = self.expand_a(x)
        b = self.expand_b(x)
        x = swish(a, b)
        x = self.squeeze(x)
        return x


class TriangleMultiplication(nn.Module):
    """Triangular multiplicative update (Algorithms 12 and 13).

    Parameters
    ----------
    d_pair: int
        Dimension of pair representation.
    d_hidden: int
        Dimension of hidden layer.
    outgoing: bool
        Whether to use outgoing edges.
    """

    def __init__(
        self,
        d_pair: int = 128,
        d_hidden: int = 128,
        outgoing: bool = True,
    ):
        super().__init__()
        self.outgoing = outgoing

        self.ln_pair = nn.LayerNorm(d_pair)
        self.to_left = Linear(d_pair, d_hidden, False, init="default")
        self.to_right = Linear(d_pair, d_hidden, False, init="default")
        self.to_gate_left = Linear(d_pair, d_hidden, False, init="gating")
        self.to_gate_right = Linear(d_pair, d_hidden, False, init="gating")

        self.ln_out = nn.LayerNorm(d_hidden)
        self.to_out = Linear(d_hidden, d_pair, False, init="final")
        self.to_gate_out = Linear(d_pair, d_pair, False, init="gating")

    def forward(
        self,
        pair: torch.Tensor,  # (B, L, L, d_pair)
        mask_2d: Optional[torch.Tensor] = None,  # (B, L, L, d_pair)
    ) -> torch.Tensor:  # (B, L, L, d_pair)
        _, L, _, _ = pair.shape

        # Compute left and right transformations
        pair = self.ln_pair(pair)
        left = self.to_left(pair)
        right = self.to_right(pair)
        left_gate = self.to_gate_left(pair)
        right_gate = self.to_gate_right(pair)
        left = sigmoid_gate(left_gate, left)
        right = sigmoid_gate(right_gate, right)

        # Mask out
        if mask_2d is not None:
            left.masked_fill_(~mask_2d[..., None], 0)
            right.masked_fill_(~mask_2d[..., None], 0)

        if self.outgoing:
            out = torch.einsum("bikd,bjkd->bijd", left, right.div_(L))
        else:
            out = torch.einsum("bkid,bkjd->bijd", left, right.div_(L))

        # Apply the gate and project back to the original dimension
        out = self.ln_out(out)
        out = self.to_out(out)
        gate = self.to_gate_out(pair)
        out = sigmoid_gate(gate, out)
        return out


class TriangleAttention(nn.Module):
    """Triangular gated self-attention (Algorithms 14 and 15).

    Parameters
    ----------
    d_pair: int
        Dimension of pair representation.
    d_hidden: int
        Dimension of hidden layer.
    n_head: int
        Number of attention heads.
    starting: bool
        Whether the attention is around the "starting" node.
    use_deepspeed: bool
        Whether to use DeepSpeed EvoformerAttention.
    """

    def __init__(
        self,
        d_pair: int = 128,
        d_hidden: int = 32,
        n_head: int = 4,
        starting: bool = True,
        use_deepspeed: bool = False,
        use_self_attention: bool = True,
    ):
        super().__init__()
        self.n_head = n_head
        self.starting = starting
        self.use_self_attention = use_self_attention
        self.use_deepspeed = False  # use_deepspeed
        if use_deepspeed and not is_deepspeed_available:
            raise ValueError("DeepSpeed is not available.")

        self.ln_pair = nn.LayerNorm(d_pair)
        self.scaling = 1 / math.sqrt(d_hidden)

        if use_self_attention:
            self.to_query = Linear(d_pair, d_hidden * n_head, False, init="glorot")
            self.to_key = Linear(d_pair, d_hidden * n_head, False, init="glorot")
        self.to_value = Linear(d_pair, d_hidden * n_head, False, init="glorot")
        self.to_bias = Linear(d_pair, n_head, False, init="default")
        self.to_gate = Linear(d_pair, n_head, False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_pair, False, init="final")

    def forward(
        self,
        pair: torch.Tensor,  # (B, L, L, d_pair)
        mask_2d: Optional[torch.Tensor] = None,  # (B, L, L)
    ) -> torch.Tensor:  # (B, L, L, d_pair)
        B, L, _, _ = pair.shape

        # Compute QKV and bias
        pair = self.ln_pair(pair)
        if self.use_self_attention:
            query = self.to_query(pair).view(B, L, L, self.n_head, -1)
            key = self.to_key(pair).view(B, L, L, self.n_head, -1)
        value = self.to_value(pair).view(B, L, L, self.n_head, -1)
        bias = self.to_bias(pair)

        if self.use_deepspeed:
            if not self.use_self_attention:
                raise NotImplementedError
            # Convert to half precision for compatibility
            query = query.half()
            key = key.half()
            value = value.half()
            bias = bias.half().permute(0, 3, 1, 2).unsqueeze(1)
            if mask_2d is not None:
                mask_2d = mask_2d.half()

            if self.starting:
                if mask_2d is not None:
                    mask_2d = mask_2d.unsqueeze(2).unsqueeze(2)

                # Compute attention
                out = DS4Sci_EvoformerAttention(query, key, value, [mask_2d, bias])
                out = out.float()
            else:
                # Transpose dimensions for "ending" attention
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                bias = bias.transpose(-1, -2)
                if mask_2d is not None:
                    mask_2d = mask_2d.transpose(1, 2).unsqueeze(2).unsqueeze(2)

                # Compute attention
                out = DS4Sci_EvoformerAttention(query, key, value, [mask_2d, bias])
                out = out.float()
                out = out.transpose(1, 2)
        else:
            if mask_2d is not None:
                mask = mask_2d.any(-1)
            if self.starting:
                bias = bias.unsqueeze(1)
                if self.use_self_attention:
                    query.mul_(self.scaling)
                    attention = torch.einsum("bijhd,bikhd->bijkh", query, key)
                    attention = attention + bias
                else:
                    attention = bias
                if mask_2d is not None:
                    attention.masked_fill_(~mask[:, None, None, :, None], float("-inf"))
                attention = F.softmax(attention, dim=-2)
                out = torch.einsum("bijkh,bikhd->bijhd", attention, value)
            else:
                bias = bias.transpose(1, 2).unsqueeze(2)
                if self.use_self_attention:
                    query.mul_(self.scaling)
                    attention = torch.einsum("bijhd,bkjhd->bijkh", query, key)
                    attention = attention + bias
                else:
                    attention = bias
                if mask_2d is not None:
                    attention.masked_fill_(~mask[:, None, None, :, None], float("-inf"))
                attention = F.softmax(attention, dim=-2)
                out = torch.einsum("bijkh,bkjhd->bijhd", attention, value)

        # Apply gate and project back to the original dimension
        gate = self.to_gate(pair).unsqueeze(-1)
        out = sigmoid_gate(gate, out)
        out = out.view(B, L, L, -1)
        out = self.to_out(out)
        return out


class OuterProduct(nn.Module):
    """Outer product single represetation to pair representation.

    Parameters
    ----------
    d_single: int
        Dimension of single representation.
    d_pair: int
        Dimension of pair representation.
    d_hidden: int
        Dimension of hidden layer.
    """

    def __init__(self, d_single: int, d_pair: int, d_hidden: int = 32):
        super().__init__()

        self.ln_single = nn.LayerNorm(d_single)
        self.to_left = Linear(d_single, d_hidden, False, init="default")
        self.to_right = Linear(d_single, d_hidden, False, init="default")
        self.to_out = Linear(d_hidden * d_hidden, d_pair, True, init="final")

    def forward(
        self,
        single: torch.Tensor,  # (B, L, d_single)
    ) -> torch.Tensor:  # (B, L, L, d_pair)
        B, L, _ = single.shape

        # Compute left and right transformations
        single = self.ln_single(single)
        left = self.to_left(single)
        right = self.to_right(single)

        # Compute outer product
        out = torch.einsum("bid,bje->bijde", left, right)
        out = out.reshape(B, L, L, -1)
        pair = self.to_out(out)
        return pair


# Linear module from OpenFold
# https://github.com/aqlaboratory/openfold/blob/815a042c697ea589d6b17180b66e9a0fc0ce041a/openfold/model/primitives.py#L692
class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


class Dropout(nn.Module):
    # Dropout entire row or column
    def __init__(self, broadcast_dim=None, p_drop=0.15):
        super().__init__()
        # give ones with probability of 1-p_drop / zeros with p_drop
        self.sampler = torch.distributions.bernoulli.Bernoulli(
            torch.tensor([1 - p_drop])
        )
        self.broadcast_dim = broadcast_dim
        self.p_drop = p_drop

    def forward(self, x):
        if not self.training:  # no drophead during evaluation mode
            return x
        shape = list(x.shape)
        if self.broadcast_dim is not None:
            shape[self.broadcast_dim] = 1
        mask = self.sampler.sample(shape).to(x.device).view(shape)

        x = mask * x / (1.0 - self.p_drop)
        return x


# TODO: move into model & change nn.Module
# Original code from Stable Diffusion 2
# https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/modules/diffusionmodules/util.py#L161
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Parameters
    ----------
    timesteps : torch.FloatTensor
        A 1-D Tensor of N indices, one per batch element. Please keep in mind
        this value is ranged from 0 to 1, so we modify the multiply factor to
        timesteps * max_period.
    dim : int
        The dimension of the output.
    max_period : int
        Controls the minimum frequency of the embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    timesteps = timesteps * max_period  # modify the multiply factor
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
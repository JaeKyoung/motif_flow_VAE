import torch
import torch.nn as nn
from typing import Optional

from motifflow.modules.primitives import (
    Transition,
    TriangleAttention,
    TriangleMultiplication,
    AttentionPairBias,
    OuterProduct,
    Dropout,
)

class PairformerBlock(nn.Module):
    """Single Pairformer block
    
    Parameters
    ----------
    d_single : int, default=384
        Dimension of single representation
    d_pair : int, default=512  
        Dimension of pair representation
    d_hidden_tri_multi : int, default=128
        Hidden dimension for triangle multiplication layers
    d_hidden_tri_attention : int, default=32
        Hidden dimension for triangle attention layers
    n_head_in_tri_attention : int, default=4
        Number of heads in triangle attention
    n_head_attention : int, default=16
        Number of heads in attention pair bias
    p_drop : float, default=0.25
        Dropout probability
    use_single : bool, default=True
        Whether to use single representation
    """

    def __init__(
        self,
        d_single: int = 384,
        d_pair: int = 512,
        d_hidden_tri_multi: int = 128,
        d_hidden_tri_attention: int = 32,
        n_head_in_tri_attention: int = 4,
        n_head_attention: int = 16,
        p_drop: float = 0.25,
        use_single: bool = True,
        use_single_cond: bool = False,
        use_deepspeed: bool = False,
        use_self_attention: bool = True,
    ):
        super().__init__()

        self.use_single_cond = use_single_cond
        if self.use_single_cond and not use_single:
            raise ValueError("use_single_cond requires use_single to be True")
        
        # Outer product layer
        if self.use_single_cond:
            self.single_to_pair = OuterProduct(d_single, d_pair)

        # Dropout layers
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        # Triangle multiplication layers
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_multi,
            outgoing=True,
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_multi,
            outgoing=False,
        )

        # Triangle attention layers
        # TODO: use global config control deepspeed option
        self.tri_atten_starting = TriangleAttention(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_attention,
            n_head=n_head_in_tri_attention,
            starting=True,
            use_deepspeed=use_deepspeed,
            use_self_attention=use_self_attention,
        )
        self.tri_atten_ending = TriangleAttention(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_attention,
            n_head=n_head_in_tri_attention,
            starting=False,
            use_deepspeed=use_deepspeed,
            use_self_attention=use_self_attention,
        )

        # Transition layers
        self.transition_pair = Transition(d_pair)
        self.use_single = use_single
        if use_single:
            # Attention and transition layers for node token features
            self.pair_to_single = AttentionPairBias(
                d_single=d_single,
                d_pair=d_pair,
                n_head=n_head_attention,
                use_deepspeed=use_deepspeed,
            )
            self.transition_single = Transition(d_single)

    def forward(
        self, 
        pair: torch.Tensor, # (B, L, L, d_pair)
        single: Optional[torch.Tensor] = None, # (B, L, d_single)
        single_mask: Optional[torch.Tensor] = None, # (B, L)
    ) -> tuple[torch.Tensor, torch.Tensor]: # (B, L, L, d_pair), (B, L, d_single)
        
        # Update pair from single
        if self.use_single_cond:
            if single is None:
                raise ValueError("single representation are required")
            pair = pair + self.single_to_pair(single)

        pair_mask = (single_mask.unsqueeze(-1).bool() & single_mask.unsqueeze(-2).bool())
        # Triangle multiplication and attention operations on edge features
        pair = pair + self.drop_row(
            self.triangle_multiplication_outgoing(pair, pair_mask)
        ) # (B, L, L, d_pair)
        pair = pair + self.drop_row(
            self.triangle_multiplication_incoming(pair, pair_mask)
        ) # (B, L, L, d_pair)
        pair = pair + self.drop_row(
            self.tri_atten_starting(pair, pair_mask)
        ) # (B, L, L, d_pair)
        pair = pair + self.drop_col(
            self.tri_atten_ending(pair, pair_mask)
        ) # (B, L, L, d_pair)
        pair = pair + self.transition_pair(pair) # (B, L, L, d_pair)

        # Update signle from pair
        if self.use_single:
            # Update node token features based on edge features
            single = single + self.pair_to_single(single, pair, single_mask.bool()) # (B, L, d_single)
            single = single + self.transition_single(single) # (B, L, d_single)

        return single, pair # (B, L, d_single), (B, L, L, d_pair)


class Pairformer(nn.Module):
    """Pairformer stack consisting of multiple Pairformer blocks
    
    This module processes node and edge token features through multiple PairformerBlocks.
    """

    def __init__(
        self,
        d_single: int = 384,
        d_pair: int = 512,
        d_hidden_tri_multi: int = 128,
        d_hidden_tri_attention: int = 32,
        n_head_in_tri_attention: int = 4,
        n_head_attention: int = 16,
        n_block_pairformer: int = 4,
        p_drop: float = 0.25,
        use_single: bool = True,
    ):
        super().__init__()
        self.n_block_pairformer = n_block_pairformer
        self.pairformer_blocks = nn.ModuleList(
            [
                PairformerBlock(
                    d_single=d_single,
                    d_pair=d_pair,
                    d_hidden_tri_multi=d_hidden_tri_multi,
                    d_hidden_tri_attention=d_hidden_tri_attention,
                    n_head_in_tri_attention=n_head_in_tri_attention,
                    n_head_attention=n_head_attention,
                    p_drop=p_drop,
                    use_single=use_single,
                )
                for _ in range(n_block_pairformer)
            ]
        )

    def forward(
        self,
        pair: torch.Tensor,  # (B, L, L, d_pair)
        single: Optional[torch.Tensor] = None,  # (B, L, d_single)
        single_mask : Optional[torch.Tensor] = None, # (B, L)
    ) -> tuple[torch.Tensor, torch.Tensor]: # (B, L, d_single), (B, L, L, d_pair)
        """Forward pass through Pairformer stack

        Parameters
        ----------
        single : torch.Tensor, optional
            Node features (B, L, d_single)
        pair : torch.Tensor, optional  
            Edge features (B, L, L, d_pair)
        single_mask : torch.Tensor, optional
            Mask for node features (B, L)
        pair_mask : torch.Tensor, optional
            Mask for edge features (B, L, L)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - Updated node features (B, L, d_single)
            - Updated edge features (B, L, L, d_pair)
        """
        for block in self.pairformer_blocks:
            single, pair = block(
                pair,
                single, 
                single_mask
            ) # (B, L, d_single), (B, L, L, d_pair)

        return single, pair
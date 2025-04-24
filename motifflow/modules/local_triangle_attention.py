import torch
from torch import nn
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from openfold.model.primitives import Attention
from openfold.model.dropout import DropoutRowwise, DropoutColumnwise
from torch import einsum
from motifflow.modules.ipa_pytorch import permute_final_dims
from motifflow.modules.ipa_pytorch import Linear
from motifflow.modules.ipa_pytorch import contact_map_gating

class LocalTriangleAttentionNew(nn.Module):
    """
    Code from Proteus [https://github.com/Wangchentong/Proteus/blob/master/model/ipa_pytorch.py]
    """
    def __init__(
            self,
            use_contact_gating,
            contact_gating_trainable,
            use_contact_score_modulation,
            contact_scale,
            c_s,
            c_z,
            c_rbf,
            c_gate_s,
            c_hidden,
            c_hidden_mul,
            no_heads,
            transition_n,
            k_neighbour,
            k_linear,
            inf,
            pair_dropout,
            **kwargs,
        ):
        super(LocalTriangleAttentionNew, self).__init__()

        self.use_contact_gating = use_contact_gating
        if self.use_contact_gating:
            self.contact_gating = contact_map_gating(contact_dim=1, c_z=c_z, trainiable_param=contact_gating_trainable)
        
        self.use_contact_score_modulation = use_contact_score_modulation
        if self.use_contact_score_modulation:
            self.contact_modulation_proj = Linear(1, no_heads, bias=False, init="final")
            # self.contact_modulation_norm = nn.LayerNorm(no_heads)
            self.contact_scale = float(contact_scale)
        
        self.embed_size = ( c_s // 2 ) * 2 + c_z
        self.no_heads = no_heads
        self.c_hidden = c_hidden
        self.c_rbf = c_rbf
        self.k_neighbour = k_neighbour
        self.k_linear = k_linear
        self.inf = inf
        self.NM_TO_ANG_SCALE = 10.0

        self.proj_left = Linear(c_s, c_gate_s)
        self.proj_right = Linear(c_s, c_gate_s)
        self.to_gate = Linear(c_gate_s*c_gate_s, c_z,init="gating")
        
        self.emb_rbf = nn.Linear(c_rbf, c_z)
        self.to_bias = Linear(c_z, self.no_heads, bias=False, init="normal")
        
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z,c_hidden_mul,)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z,c_hidden_mul,)
        
        self.mha_start = Attention(c_z, c_z, c_z, self.c_hidden, self.no_heads)
        self.mha_end = Attention(c_z, c_z, c_z, self.c_hidden, self.no_heads)
        
        # self.pair_transition = PairTransition(c_z, transition_n,)
        
        self.dropout_row_layer = DropoutRowwise(pair_dropout)
        self.dropout_col_layer = DropoutColumnwise(pair_dropout)
       
        self.layer_norm = nn.LayerNorm(c_z)

    def local_mha(self, x, rigids, num_neighbour, num_linear, triangle_bias, contact_modulation_full, mask, starting_node):
        '''
        Args:
            x: [batch_size, residue_num, residue_num, embed_size]
            rigids: [batch_size, residue_num, 3, 3]
            num_neighbour: int
            num_linear: int
            triangle_bias: [batch_size, residue_num, residue_num, num_heads]
            mask: [batch_size, residue_num, residue_num]
            starting_node: bool
        Returns:
            x: [batch_size, residue_num, residue_num, embed_size]
        '''
        B, N, _, D = x.size()
        B, N, _, H = triangle_bias.size()
        
        out_x = torch.zeros_like(x)
        coords = rigids.get_trans()
        
        if not starting_node:
            x = x.transpose(-2, -3)
            triangle_bias = triangle_bias.transpose(-2, -3)
            if contact_modulation_full is not None:
                 contact_modulation_full = contact_modulation_full.transpose(-2,-3)
            mask = mask.transpose(-1, -2)
            
        # [batch_size, residue_num, num_neighbour]
        indices = self.knn_indices(coords, num_neighbour,num_linear, pair_mask=mask)
        num_neighbour = num_neighbour + num_linear
        
        x = torch.gather(
            x, dim=2, index=indices.unsqueeze(-1).expand(B, N, num_neighbour, D)
        )
        x = self.layer_norm(x)
        # [B, I, K]
        mask = torch.gather(
            mask, dim=2, index=indices
        )
        # [B, I, 1, 1, K]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        
        triangle_bias = triangle_bias.unsqueeze(-2).expand(B, N, N, N, H,) # [B, I, J, K, H]
        triangle_bias = torch.gather(triangle_bias, dim=2, index=indices[...,None,None].expand((B,N,num_neighbour,N,H,))) # [B, I, k, K, H]
        triangle_bias = torch.gather(triangle_bias, dim=3, index=indices[...,None,:,None].expand((B,N,num_neighbour,num_neighbour,H,))) # [B, I, k, k, H]
        triangle_bias = permute_final_dims(triangle_bias, (2, 1, 0)) # [B, I, H, k, k]
        
        biases = [mask_bias, triangle_bias]

        if contact_modulation_full is not None:
            # contact_modulation_full: [B, N, N, H]
            contact_modulation_local_expanded = contact_modulation_full.unsqueeze(-2).expand(B, N, N, N, H) # [B, N, K_total, N, H]
            contact_modulation_local = torch.gather(contact_modulation_local_expanded, dim=2, index=indices.view(B, N, 1, num_neighbour, 1).expand(-1, -1, N, -1, H)) # [B, I, k, K, H]
            contact_modulation_local = torch.gather(contact_modulation_local, dim=3, index=indices.view(B, N, 1, num_neighbour, 1).expand(B, N, num_neighbour, num_neighbour, H)) # [B, N, K_total, K_total, H]
            contact_modulation_local = permute_final_dims(contact_modulation_local, (2, 1, 0)) # [B, N, H, K_total, K_total]
            # LayerNorm
            # contact_modulation_local_permuted = contact_modulation_local.permute(0, 1, 3, 4, 2) # [B, N, K, K, H]
            # contact_modulation_local_norm = self.contact_modulation_norm(contact_modulation_local_permuted)
            # contact_modulation_local = contact_modulation_local_norm.permute(0, 1, 4, 2, 3) # 다시 [B, N, H, K, K]
            
            scaled_contact_modulation = self.contact_scale * contact_modulation_local
            biases.append(scaled_contact_modulation)
        
        if starting_node:
            x = self.mha_start(q_x=x, kv_x=x, biases=biases)
        else:
            x = self.mha_end(q_x=x, kv_x=x, biases=biases)
            
        out_x = out_x.scatter(2, indices.unsqueeze(-1).expand(B, N, num_neighbour, D), x)
        
        if not starting_node:
            out_x = out_x.transpose(-2, -3)
            
        return out_x
    
    def knn_indices(self, x, num_neighbour, num_linear, pair_mask = None):
        _,nres = x.shape[:2]
        
        # Warning : we advise to use this commented no bug line for new model, we left the buggy line to keep with the original trained model.
        # distances = torch.norm(x.unsqueeze(2) - x.unsqueeze(1), dim=-1) * self.NM_TO_ANG_SCALE
        distances = torch.norm(x.unsqueeze(2) - x.unsqueeze(1), dim=-1)
        distances[:, torch.arange(0, nres, dtype=torch.long), torch.arange(0, nres, dtype=torch.long)] = self.inf
        # eye_mask = torch.eye(nres, dtype=torch.bool, device=x.device)
        # distances.masked_fill_(eye_mask, self.inf)
        
        # set distance between linear neighbour to 0
        for i in range(1,num_linear//2+1):
            row_indices = torch.arange(0, nres, dtype=torch.long)
            indices = torch.arange(i, nres+i, dtype=torch.long)
            distances[:, row_indices[:nres-i], indices[:nres-i]] = 0
            indices = torch.arange(i*-1, nres-i, dtype=torch.long)
            distances[:, row_indices[i:], indices[i:]] = 0
        
        if pair_mask is not None:
            distances = distances + (self.inf * (pair_mask - 1))

        _, indices = torch.topk(distances, num_neighbour+num_linear, dim=-1, largest=False)  # Shape: [B, N, K]
        return indices

    def rbf(self, D, D_min=0.0, D_sigma=0.5):
        # Distance radial basis function
        D_max = D_min + (self.c_rbf-1) * D_sigma
        D_mu = torch.linspace(D_min, D_max, self.c_rbf).to(D.device)
        D_mu = D_mu[None,:]
        D_expand = torch.unsqueeze(D, -1)
        rbf_feat = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        rbf_feat = self.emb_rbf(rbf_feat)
        return rbf_feat

    def forward(self, node_embed, edge_embed, rigids, edge_mask, contact_map, t):
        
        batch_size, num_res, _ = node_embed.shape
        device = node_embed.device

        if self.use_contact_gating:
            edge_embed = self.contact_gating(edge_embed, contact_map, t)

        contact_modulation_full = None
        if self.use_contact_score_modulation:
            t_tensor = t
            gate = (1.0 - t_tensor).view(-1, 1, 1, 1) # [B, 1, 1, 1]
            gated_p_contact = gate * contact_map.unsqueeze(-1) # [B, N, N, 1]
            contact_modulation_full = self.contact_modulation_proj(gated_p_contact) # [B, N, N, H]

        # get pair bias from rbf of distance
        coords = rigids.get_trans()
        distances = torch.norm(coords.unsqueeze(2) - coords.unsqueeze(1), dim=-1)
        bias = self.rbf(distances)
        
        # gate pair bias with sequence embedding
        left = self.proj_left(node_embed)
        right = self.proj_right(node_embed)
        gate = einsum('bli,bmj->blmij', left, right).reshape(batch_size,num_res,num_res,-1)
        gate = torch.sigmoid(self.to_gate(gate))
        bias = bias * gate
        # pair bias shape : [B,N,N,h]
        bias = self.to_bias(bias)
        
        z = edge_embed
        z = z + self.dropout_row_layer(self.tri_mul_out(z, mask=edge_mask))
        z = z + self.dropout_row_layer(self.tri_mul_in(z, mask=edge_mask))
        z = z + self.dropout_row_layer(self.local_mha(z, rigids, self.k_neighbour, self.k_linear, triangle_bias=bias, contact_modulation_full=contact_modulation_full, mask=edge_mask, starting_node=True))
        z = z + self.dropout_col_layer(self.local_mha(z, rigids, self.k_neighbour, self.k_linear, triangle_bias=bias, contact_modulation_full=contact_modulation_full, mask=edge_mask, starting_node=False))

        # Pair transition
        # z = self.pair_transition(z)

        return z
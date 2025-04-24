import torch
from torch import nn
from motifflow.modules.utils import get_index_embedding, get_time_embedding
from motifflow.data.residue_constants import restypes_with_x

class NodeFeatureNet(nn.Module):
    """
    Node feature network.
    Generates node features for a graph neural network.
    """
    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s 
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_chain_emb = self._cfg.c_chain_emb
        self.max_n_res = self._cfg.max_n_res
        self.max_n_chain = self._cfg.max_n_chain

        # Calculate total embedding size
        embed_size = self.c_pos_emb + self.c_chain_emb + 21 + self.c_timestep_emb * 2 + 3
        if self._cfg.use_z_all:
            self.c_latent_emb = self._cfg.c_latent_emb
            embed_size += self.c_latent_emb
        
        # Layer for final projection    
        self.linear = nn.Linear(embed_size, self.c_s, bias=False)
    
    def embed_t(self, timesteps, num_res, residue_mask):
        """
        Embed the timesteps.
        """
        # Generate time embeddings for each batch and repeat for all residues
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=self._cfg.n_timestep
        )[:, None, :].repeat(1, num_res, 1)
        return timestep_emb * residue_mask.unsqueeze(-1)

    def forward(
        self,
        so3_t: torch.Tensor,           # SO(3) timestep tensor
        r3_t: torch.Tensor,            # R3 timestep tensor
        residue_index: torch.Tensor,   # Residue indices
        residue_mask: torch.Tensor,    # Residue mask
        chain_index: torch.Tensor,     # Chain indices
        scaffold_mask: torch.Tensor,   # Scaffold mask
        motif_mask: torch.Tensor,      # Motif mask
        aatype: torch.Tensor,          # Amino acid types
        interface_mask: torch.Tensor,  # Interface mask
        z_all: torch.Tensor            # Latent embedding
    ):
        b, num_res, device = residue_mask.shape[0], residue_mask.shape[1], residue_mask.device

        # Positional resdiue index embedding [B, N, c_pos_emb]
        # pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(residue_index, self.c_pos_emb, self.max_n_res) * residue_mask.unsqueeze(-1)

        # Chain index embedding [B, N, c_chain_emb]
        chain_emb = get_index_embedding(chain_index, self.c_chain_emb, self.max_n_chain) * residue_mask.unsqueeze(-1)
        
        # Timestep embedding
        # [B, N, c_timestep_emb]
        so3_t_emb = self.embed_t(so3_t, num_res, residue_mask)
        r3_t_emb = self.embed_t(r3_t, num_res, residue_mask) 

        # Masks [B, N, 1]
        # Prepare masks and aatype embedding
        scaffold_mask = scaffold_mask.unsqueeze(-1)
        motif_mask = motif_mask.unsqueeze(-1)
        interface_mask = interface_mask.unsqueeze(-1)

        # one-hot aatype embedding 
        # make [B,N] integer (0~20 A to X) to [B, N, 21] one-hot vector
        if self.training and self._cfg.seq_dropout > 0:
            strategy_prob = torch.rand(1, device=device).item()
            motif_mask_bool = motif_mask.squeeze(-1).bool()
            chain_1_mask = (chain_index == 1)
            chain_1_motif_mask = motif_mask_bool & chain_1_mask
            alanine_id = restypes_with_x.index('A')

            # dropout if dropout_mask is False  
            dropout_mask = torch.ones_like(aatype, dtype=torch.bool, device=device)

            if strategy_prob < 0.15:
                dropout_mask[chain_1_motif_mask] = False
            elif strategy_prob < 0.50:
                dropout_mask[chain_1_motif_mask] = torch.rand_like(
                    aatype[chain_1_motif_mask], dtype=torch.float32
                ) > self._cfg.seq_dropout
            else:
                dropout_mask[chain_1_motif_mask] = True
                
            aatype = torch.where(dropout_mask, aatype, torch.tensor(alanine_id, device=device))

        # aatype_onehot = torch.nn.functional.one_hot(aatype, num_classes=len(restypes))
        aatype_onehot = torch.nn.functional.one_hot(aatype, num_classes=len(restypes_with_x))
        aatype_emb = aatype_onehot * motif_mask
        
        z_feat = [z_all] if self._cfg.use_z_all else []
        all_node_feats = [
            pos_emb,             # [B, N, c_pos_emb]
            chain_emb,           # [B, N, c_chain_emb]
            aatype_emb,          # [B, N, 21]
            so3_t_emb,           # [B, N, c_timestep_emb]
            r3_t_emb,            # [B, N, c_timestep_emb]
            *z_feat,             # [B, N, c_latent_emb]
            motif_mask,          # [B, N, 1]
            motif_mask,          # [B, N, 1]
            interface_mask,      # [B, N, 1]
        ]

        # Generate node features [B, N, c_s]
        node_features = self.linear(torch.cat(all_node_feats, dim=-1))
        return node_features
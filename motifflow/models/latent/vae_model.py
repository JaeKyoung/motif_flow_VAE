import torch
import torch.nn as nn
import torch.nn.functional as F
from motifflow.data import all_atom
from motifflow.models.latent.utils import (
    ProteinmotifFeatures,
    ProteinMPNN_EncLayer,
    ProteinMPNN_DecLayer,
    gather_nodes,
    cat_neighbors_nodes,
)
from motifflow.modules.utils import cal_distogram, cal_unit_vector
from motifflow.modules.utils import get_index_embedding
import motifflow.data.utils as du
from motifflow.modules.primitives import RelativePositionEmbedding, Transition, Linear
from motifflow.modules.pairformer import Pairformer

# TODO : NVAE, cyclical_annealing, encoder with pairformer

#########################
# VAEEncoder Definition
#########################
class VAEEncoder(nn.Module):
    def reparameterize(self, mean, logvar):
        # Reparameterization trick: z = mu + eps * std, where eps ~ N(0, I).
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, trans_1, rotmats_1, aatype, motif_mask, residue_mask, residue_index, chain_index): ...

    def forward(self, trans_1, rotmats_1, aatype, motif_mask, residue_mask, residue_index, chain_index):
        z, mean, logvar = self.encode(trans_1, rotmats_1, aatype, motif_mask, residue_mask, residue_index, chain_index)
        return z, mean, logvar

class Encoder_Transformer(VAEEncoder):
    """
    Encoder that embeds protein backbone coordinates (and amino acid types) into a latent space with pairformer.
    
    Inputs:
        trans_1: (B, N, 3) Translation coordinates of protein structure
        rotmats_1: (B, N, 3, 3) Rotation matrices of protein structure  
        aatype: (B, N) Amino acid type indices
        motif_mask: (B, N) Motif mask
        residue_mask: (B, N) Residue mask
        residue_index: (B, N) Residue indices
        chain_index: (B, N) Chain indices
        
    Outputs:
        z: (B, N, latent_dim) Encoded latent representation
        mean: (B, N, latent_dim) Mean values for each residue
        logvar: (B, N, latent_dim) Log variance values for each residue
    
    Configuration (cfg):
        - hidden_dim: Transformer hidden dimension size
        - latent_dim: VAE latent dimension size 
        - c_pos_emb: Position embedding dimension
        - c_chain_emb: Chain embedding dimension
        - max_n_res: Maximum number of residues
        - max_n_chain: Maximum number of chains
        - dgram_bins: Number of distance gram bins
        - relpos_bins: Number of relative position bins
        - num_layers: Number of pairformer blocks
    """
    def __init__(self, cfg):
        super().__init__()
        self.encoder_cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.latent_dim = cfg.latent_dim
        
        embed_size = self.encoder_cfg.c_pos_emb + self.encoder_cfg.c_chain_emb + 3
        self.linear_single = nn.Linear(embed_size, 2 * self.hidden_dim, bias=False)
        self.transition_single = Transition(2 * self.hidden_dim)
        self.dgram_bins = cfg.dgram_bins
        self.relpos_bins = cfg.relpos_bins
        self.relpos_embedder = RelativePositionEmbedding(self.hidden_dim, self.relpos_bins)
        d_concat = self.dgram_bins + 3 + self.hidden_dim
        self.linear_cat = Linear(d_concat, self.hidden_dim, False)
        self.transition_pair = Transition(self.hidden_dim)
        self.pairformer_blocks = Pairformer(
            d_single=2 * self.hidden_dim,
            d_pair=self.hidden_dim,
            n_block_pairformer=cfg.num_layers,
        )

        self.latent_projector = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)
    
    def encode(self, trans_1, rotmats_1, aatype, motif_mask, residue_mask, residue_index, chain_index):
        batch_size, num_res = trans_1.shape[:2]
        device = trans_1.device
        motif_mask_unsqueezed = motif_mask.unsqueeze(-1)
        motif_mask_unsqueezed_2d = motif_mask.unsqueeze(-1).unsqueeze(-1)
        
        trans_1 = trans_1 * motif_mask_unsqueezed
        rotmats_1 = rotmats_1 * motif_mask_unsqueezed_2d
        aatype = aatype * motif_mask

        # Get Single Representation
        pos_emb = get_index_embedding(residue_index, self.encoder_cfg.c_pos_emb, self.encoder_cfg.max_n_res) * residue_mask.unsqueeze(-1)
        chain_emb = get_index_embedding(chain_index, self.encoder_cfg.c_chain_emb, self.encoder_cfg.max_n_chain) * residue_mask.unsqueeze(-1)
        interface_mask = torch.zeros_like(motif_mask_unsqueezed)
        single = torch.cat([pos_emb, chain_emb, motif_mask_unsqueezed, motif_mask_unsqueezed, interface_mask], dim=-1)
        single = self.linear_single(single)
        single = self.transition_single(single) * motif_mask_unsqueezed # (B, N, 2 * hidden_dim)

        # Get Pair Representaataion
        dgram = cal_distogram(trans_1, num_bins=self.dgram_bins) # (B, N, N, dgram_bins)
        unitvec = cal_unit_vector(du.create_rigid(rotmats_1, trans_1)) # (B, N, N, 3)
        relpos = self.relpos_embedder(residue_index) # (B, N, N, hidden_dim)
        pair = torch.cat([dgram, unitvec, relpos], dim=-1)
        pair = self.linear_cat(pair)
        pair = pair + self.transition_pair(pair) * motif_mask_unsqueezed_2d # (B, N, N, hidden_dim)

        # Update pair and get single representation
        single, _ = self.pairformer_blocks(single=single, pair=pair, single_mask=motif_mask) # (B, N, 2 * hidden_dim)
        # single = pair[:, :, 0]
        single = single * motif_mask_unsqueezed
        latent = self.latent_projector(single) * motif_mask_unsqueezed

        mean = self.fc_mean(latent) * motif_mask_unsqueezed
        logvar = self.fc_logvar(latent) * motif_mask_unsqueezed
        z_all = self.reparameterize(mean, logvar) * motif_mask_unsqueezed

        return z_all, mean, logvar

class Encoder_MPNN(VAEEncoder):
    """
    ProteinMPNN-inspired encoder that embeds backbone coordinates and amino acid types.
    
    Inputs:
        NCaCO_pos: (B, N, 4, 3) backbone coordinates [N, CA, C, O]
        aatype: (B, N) amino acid type indices
        motif_mask: (B, N) mask for motif residues
        residue_mask: (B, N) mask
        residue_index: (B, N) residue indices
        chain_index: (B, N) chain indices
        
    Outputs:
        z_all: (B, N, latent_dim) encoded representation
        mean: (B, N, latent_dim) mean for each residue
        logvar: (B, N, latent_dim) log-variance for each residue
    
    Configuration (cfg):
        - hidden_dim: Hidden dimension size
        - latent_dim: Latent space dimension size
        - num_aa: Number of amino acid types
        - node_features: Node feature dimension
        - edge_features: Edge feature dimension  
        - K_neighbors: Number of nearest neighbors to consider
        - augment_eps: Noise augmentation epsilon
        - dropout: Dropout probability
        - num_layers: Number of encoder layers
    """
    def __init__(self, cfg):
        super().__init__()
        self.encoder_cfg = cfg
        self.latent_dim = cfg.latent_dim

        self.node_features = cfg.node_features
        self.edge_features = cfg.edge_features
        self.hidden_dim = cfg.hidden_dim
        self.num_layers = cfg.num_layers
        self.K_neighbors = cfg.K_neighbors
        self.augment_eps = cfg.augment_eps
        
        self.motiffeatures = ProteinmotifFeatures(
            edge_features=self.edge_features,
            node_features=self.node_features,
            top_k=self.K_neighbors,
            augment_eps=self.augment_eps
        )
        self.W_e = nn.Linear(self.edge_features, self.hidden_dim, bias=True)
        
        self.pMPNN_encoder_layers = nn.ModuleList([ProteinMPNN_EncLayer(self.hidden_dim, self.hidden_dim * 2, dropout=cfg.dropout) for _ in range(self.num_layers // 2)])
        self.pMPNN_decoder_layers = nn.ModuleList([ProteinMPNN_DecLayer(self.hidden_dim, self.hidden_dim * 3, dropout=cfg.dropout) for _ in range(self.num_layers // 2)])
        
        self.final_layer = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_mean = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.constant_(self.fc_logvar.bias, -1.0)


    def encode(self, trans_1, rotmats_1, aatype, motif_mask, residue_mask, residue_index, chain_index):
        
        all_bb_pos = all_atom.atom37_from_trans_rot(trans_1, rotmats_1, residue_mask)
        NCaCO_pos = all_bb_pos[..., [0, 1, 2, 4], :]
        
        B, num_res = NCaCO_pos.shape[:2]
        device = NCaCO_pos.device

        motif_mask_unsqueezed = motif_mask.unsqueeze(-1)
        motif_mask_unsqueezed_2d = motif_mask.unsqueeze(-1).unsqueeze(-1)
        
        NCaCO_pos = NCaCO_pos * motif_mask_unsqueezed_2d # (B, N, 4, 3)
        aatype = aatype * motif_mask # (B, N)
        residue_index = residue_index * motif_mask # (B, N)
        chain_index = chain_index * motif_mask # (B, N)
        
        # Edge features
        E, E_idx = self.motiffeatures(NCaCO_pos, residue_index, chain_index, motif_mask) # E: (B, N, K, E), E_idx: (B, N, K)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device) # (B, N, E)
        h_E = self.W_e(E) # (B, N, K, hidden_dim) 

        mask_attend = gather_nodes(motif_mask_unsqueezed, E_idx).squeeze(-1) # (B, N, K)
        mask_attend = mask_attend * motif_mask_unsqueezed

        # ProteinMPNN Encoder Layers
        for layer in self.pMPNN_encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, motif_mask, mask_attend) # (B, N, hidden_dim), (B, N, K, hidden_dim)
        
        # ProteinMPNN Decoder for Latent Distribution
        h_V_stack = [h_V] + [
            torch.zeros_like(h_V, device=device)
            for _ in range(len(self.pMPNN_decoder_layers))
        ]
        for l, layer in enumerate(self.pMPNN_decoder_layers):
            h_V_neighbor = gather_nodes(h_V_stack[l], E_idx)  # (B, N, K, hidden_dim)
            h_V_target = h_V_stack[l].unsqueeze(2).expand(-1, -1, E_idx.shape[-1], -1)  # (B, N, K, hidden_dim)
            h_ESV = torch.cat([h_V_target, h_E, h_V_neighbor], dim=-1)  # (B, N, K, 3 * hidden_dim)
            h_V_stack[l + 1] = layer(h_V_stack[l], h_ESV, mask_V=residue_mask)
        
        # Final decoder output
        h_V_final = h_V_stack[-1]  # (B, N, hidden_dim)
        latent = F.relu(self.final_layer(h_V_final))  # (B, N, latent_dim)
        mean = self.fc_mean(latent) * motif_mask_unsqueezed
        logvar = self.fc_logvar(latent) * motif_mask_unsqueezed
        z = self.reparameterize(mean, logvar) * motif_mask_unsqueezed
        z = z *residue_mask[..., None]
        mean = mean *residue_mask[..., None]
        logvar = logvar *residue_mask[..., None]
        
        return z, mean, logvar
    


#########################
# Decoder Definition
#########################
class Decoder(nn.Module):
    """
    Decoder that reconstructs contact map from latent vectors using Kronecker-product style combination.
    
    Input:
        z: (B, N, latent_dim) latent vectors
    Output:
        contact_map: (B, N, N) contact probabilities
    """
    def __init__(self, cfg):
        super().__init__()
        self.decoder_cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.hidden_dim = cfg.hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim * self.latent_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim,  self.hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim//2, 1)
        )
        
    def decode(self, z, motif_mask, residue_mask):
        batch_size, num_res, latent_dim = z.shape
        z_i = z.unsqueeze(2)  # [B, N, 1, d]
        z_j = z.unsqueeze(1)  # [B, 1, N, d]
        outer_product = z_i.unsqueeze(-1) * z_j.unsqueeze(-2) # [B, N, N, d, d]
        kronecker = outer_product.view(batch_size, num_res, num_res, -1) # [B, N, N, d*d]
        contact_logits = self.mlp(kronecker).squeeze(-1) # [B, N, N]
        outer_mask = motif_mask.unsqueeze(1) * motif_mask.unsqueeze(2)  # [B, N, N]
        contact_logits = contact_logits * outer_mask
        contact_map = torch.sigmoid(contact_logits)  # [B, N, N]

        return contact_map, contact_logits


    def forward(self, z, motif_mask, residue_mask):
        contact_map, contact_logits = self.decode(z, motif_mask, residue_mask)
        return contact_map, contact_logits
import torch
from torch import nn
from motifflow.modules.utils import distance, rot_to_quat, get_index_embedding

class EdgeFeatureNet(nn.Module):
    """
    Edge feature network for generating edge features in a graph neural network.
    """
    def __init__(self, module_cfg):
        super(EdgeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s 
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim
        self.relpos_k = self._cfg.relpos_k
        self.relpos_n_bin = 2 * self.relpos_k + 2

        # Linear layers for various feature transformations
        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.relpos_n_bin + 1, self.c_p, bias=False)
        # self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)
        if self._cfg.use_contact_map:
            self.linear_pred_contact_map = nn.Linear(1, self.c_p, bias=False)
        self.linear_template = nn.Linear(self._cfg.template_dist_n_bin + 6, self.c_p, bias=False)
        if self._cfg.add_motif_template:
            self.linear_motif_template = nn.Linear(self._cfg.template_dist_n_bin + 2, self.c_p, bias=False)
        if self._cfg.self_condition:
            self.linear_sc = nn.Linear(self._cfg.template_dist_n_bin + 2, self.c_p, bias=False)

        # Configuration for distance binning
        self.template_dist_min = self._cfg.template_dist_min
        self.template_dist_step = self._cfg.template_dist_step
        self.template_dist_n_bin = self._cfg.template_dist_n_bin
        
        if self._cfg.type == "concatenation":
            # Calculate total dimension of edge features
            total_edge_feats = self.c_p * 5 + 2
            
            # Edge embedding network
            self.edge_embedder = nn.Sequential(
                nn.Linear(total_edge_feats, self.c_p),
                nn.ReLU(),
                nn.Linear(self.c_p, self.c_p),
                nn.ReLU(),
                nn.Linear(self.c_p, self.c_p),
                nn.LayerNorm(self.c_p),
            )

    def embed_relpos(self, r):
        # AlphaFold 2 Algorithm 4 & 5
        # Based on OpenFold utils/tensor_utils.py
        # Input: [b, n_res]
        # Output : [B, N, N, feat_dim]
        # [b, n_res, n_res]
        d = r[:, :, None] - r[:, None, :]
        pos_emb = get_index_embedding(d, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)
    
    def _relpos(self, residue_index, chain_index):
        """
        Compute relative position encoding based on residue indices (within the chain) and chain indices.

		This algorithm is adopted from AlphaFold 2 Algorithm 4 & 5 and implemented based on OpenFold utils/tensor_utils.py.

        Args:
            residue_index: [B, N] Residue indices (starting from 1)
            chain_index: [B, N] Chain indices (starting from 1)

		Returns:
			[B, N, N, c_p] Pair representation based on pairwise relative positions
		"""
        # Denotes if two residues are in the same chain
		# Shape: [B, N, N]
        is_same_chain = chain_index[:, :, None] == chain_index[:, None, :]

        # Pairwise relative position matrix, offsetted by window size
		# Note that relative residue position across chains is capped at
		# relpos_k + 1, or 2 * relpos_k + 1 with offset
		# Shape: [B, N, N]
        d_same_chain = torch.clip(residue_index[:, :, None] - residue_index[:, None, :] + self.relpos_k, 0, 2 * self.relpos_k)
        d_diff_chain = torch.ones_like(d_same_chain) * (2 * self.relpos_k + 1)
        d = d_same_chain * is_same_chain + d_diff_chain * ~is_same_chain

        # Pairwise relative position encoding
        # Shape: [B, N, N, n_bin]
        oh = nn.functional.one_hot(d.long(), num_classes=self.relpos_n_bin).float()
        rel_pos_feat = torch.cat([oh, is_same_chain.unsqueeze(-1).float()], dim=-1)

        # Project to given single representation dimension
        # Shape: [B, N, N, c_p]
        return self.linear_relpos(rel_pos_feat)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        """
        Create pairwise features by concatenating 1D features.
        From FrameFlow
        """
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def _encode_positions(self, coords, mask):
        """
        Encode pairwise distances for a sequence of coordinates.

        Args:
            coords: [B, N, 3] A sequence of atom positions.
            mask: [B, N] Mask to indicate which atom position is masked.

        Returns:
            [B, N, N, n_bin] Masked pairwise distance encoding
        """
         # Pairwise distance matrix [B, N, N]
        d = distance(torch.stack([
            coords.unsqueeze(2).repeat(1, 1, coords.shape[1], 1),
            coords.unsqueeze(1).repeat(1, coords.shape[1], 1, 1),
        ], dim=-2))

        # Distance bins [n_bin]
        v = torch.arange(0, self.template_dist_n_bin, device=coords.device)
        v = self.template_dist_min + v * self.template_dist_step
        
        # Reshaped distance bins [1, 1, 1, n_bin]
        v_reshaped = v.view(*((1,) * len(d.shape) + (len(v),)))
        
        # Pairwise distance bin matrix [B, N, N]
        b = torch.argmin(torch.abs(d.unsqueeze(-1) - v_reshaped), dim=-1)
        
        # Pairwise distance bin encoding [B, N, N, n_bin]
        oh = nn.functional.one_hot(b, num_classes=len(v)).float()

        # Pairwise mask [B, N, N]
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        return oh * pair_mask.unsqueeze(-1)

    def _encode_orientations(self, rots, mask):
        """
        Encode pairwise relative orientations for a sequence of frames.

        Args:
            rots: [B, N, 3, 3] A sequence of orientations.
            mask: [B, N] Mask to indicate which orientation is masked.

        Returns:
            [B, N, N, 4] Masked pairwise relative orientation encoding (quaternions)
        """
        # Pairwise rotation matrix  [B, N, N, 3, 3]
        r = torch.matmul(rots.unsqueeze(1),rots.unsqueeze(2))

        # Pairwise quaternion [B, N, N, 4]
        q = rot_to_quat(r)
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        return q * pair_mask.unsqueeze(-1)
    
    def forward(
        self,
        t,                      # Time step
        s,                      # Single representation [B, N, c_s]
        edge_mask,              # Edge mask
        residue_index,          # Residue indices
        chain_index,            # Chain indices  
        residue_mask,           # Residue mask
        trans_1,                # Initial translations
        rotmats_1,              # Initial rotation matrices
        trans_t,                # Target translations
        rotmats_t,              # Target rotation matrices
        trans_sc,               # Self-conditioning translations
        rotmats_sc,             # Self-conditioning rotation matrices
        scaffold_mask,          # Scaffold mask
        motif_mask,             # Motif mask
        fixed_structure_mask,    # Fixed structure mask
        pred_contact_map,        # Predicted contact map
    ):
        
        # input single representation : [B, N, c_s]
        num_batch, num_res, _ = s.shape

        # Generate various features
        # cross_node_features 
        # [B, N, N, 2feat_dim=c_p] by frameflow
        # [1, 256, 256, 128]
        cross_node_feats = self._cross_concat(self.linear_s_p(s), num_batch, num_res)
        
        # Relative Position and Chain embedding 
        # [B, N, N, c_p] following genie2
        # residue_index [1, 256]
        # chain_index [1,256]
        relpos_feats = self._relpos(residue_index, chain_index)

        """
        # Relative Position embedding by FrameFlow 
        # [B, N]
        r = torch.arange(num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        # [B, N, N, feat_dim]
        relpos_feats_ff = self.embed_relpos(r)
        """

        # All Pairwise relative position
        # [B, N, N, c_p]
        template_feats = self.linear_template(torch.cat([
            self._encode_positions(trans_t, residue_mask),
            self._encode_orientations(rotmats_t, residue_mask),
            fixed_structure_mask.unsqueeze(-1),
            fixed_structure_mask.unsqueeze(-1)
        ], dim=-1))

        # Predicted contact map
        # [B, N, N, c_p]
        if self._cfg.use_contact_map:
            contact_map = pred_contact_map.unsqueeze(-1) # [B, N, N, 1]
            if self._cfg.contact_gating:
                gate = t.unsqueeze(-1).unsqueeze(-1)**2  # [B, 1, 1]
                gated_contact = gate * contact_map # Apply soft transform
                pred_contact_map_feats = self.linear_pred_contact_map(gated_contact)
            else:
                pred_contact_map_feats = self.linear_pred_contact_map(contact_map)

        # Motif Pairwise relative position
        # motif_feats: [B, N, N, c_p]
        if self._cfg.add_motif_template:
            motif_feats = self.linear_motif_template(torch.cat([
                self._encode_positions(trans_1, motif_mask) * fixed_structure_mask.unsqueeze(-1),
                # self._encode_orientations(rotmats_1, motif_mask) * fixed_structure_mask.unsqueeze(-1),
                fixed_structure_mask.unsqueeze(-1),
                fixed_structure_mask.unsqueeze(-1),
            ], dim=-1))
            
        # Self-conditioning
        # [B, N, N, c_p]
        template_sc_feats = self.linear_sc(torch.cat([
            self._encode_positions(trans_sc, residue_mask),
            # self._encode_orientations(rotmats_sc, residue_mask),
            fixed_structure_mask.unsqueeze(-1),
            fixed_structure_mask.unsqueeze(-1),
        ], dim=-1))
            
        # Combine all features
        if self._cfg.type == "aggregation":
            edge_feats = cross_node_feats + relpos_feats + template_feats
            if self._cfg.use_contact_map:
                edge_feats += pred_contact_map_feats
            if self._cfg.add_motif_template:
                edge_feats += motif_feats
            if self._cfg.self_condition:
                edge_feats += template_sc_feats
            edge_feats *= edge_mask.unsqueeze(-1)
            return edge_feats
        
        elif self._cfg.type == "concatenation":
            # TODO 수정하기
            # Concatenate all input features
            all_edge_feats = [
                cross_node_feats,    # [B, N, N, 2*feat_dim=c_p]
                relpos_feats,        # [B, N, N, c_p]
                template_feats,      # [B, N, N, c_p]
                motif_feats,         # [B, N, N, c_p]
                template_sc_feats,   # [B, N, N, c_p]
                fixed_structure_mask, # [B, N, N, 1]
                fixed_structure_mask, # [B, N, N, 1]
            ]

            edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
            edge_feats *= edge_mask.unsqueeze(-1)
            return edge_feats

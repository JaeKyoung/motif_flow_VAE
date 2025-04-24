import torch
import torch.nn as nn

from motifflow.modules.node_feature_net import NodeFeatureNet
from motifflow.modules.edge_feature_net import EdgeFeatureNet
from motifflow.modules.pair_transform_net import PairTransformNet
from motifflow.modules import ipa_pytorch
# from motifflow.modules import heads
from motifflow.modules import local_triangle_attention
from motifflow.data import utils as du

from openfold.model.primitives import Attention

class FlowModel(nn.Module):
    def __init__(self, structure_model_cfg):
        super(FlowModel, self).__init__()

        # Flow module
        self.structure_model_cfg = structure_model_cfg
        self._ipa_conf = self.structure_model_cfg.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 

        # Node feature net
        self.node_feature_net = NodeFeatureNet(self.structure_model_cfg.node_features)
        self.edge_feature_net = EdgeFeatureNet(self.structure_model_cfg.edge_features)

        # Pair Transform Network (triangular multiplicative updates)
        if self.structure_model_cfg.pair_transform_net.enable:
            self.pair_transform_net = PairTransformNet(**self.structure_model_cfg.pair_transform_net)

        """
        # Pairformer
        if self.model_conf.pairformer.enable:
            self.pairformer_blocks = Pairformer(**self.model_conf.pairformer)
        """

        # Main Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            
            # TODO : use vector field network?

            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)

            # Skip embedding
            if self._ipa_conf.c_skip_embedding.enable:
                self.trunk[f'skip_embed_{b}'] = ipa_pytorch.Linear(self.structure_model_cfg.node_embed_size, self._ipa_conf.c_skip_embedding.c_skip, init="final")
                tfmr_in = self._ipa_conf.c_s + self._ipa_conf.c_skip_embedding.c_skip
            else:
                tfmr_in = self._ipa_conf.c_s

            # Use pytorch or flash attention
            if self._ipa_conf.seq_tfmr_attention == "pytorch":
                tfmr_layer = torch.nn.TransformerEncoderLayer(
                    d_model=tfmr_in,
                    nhead=self._ipa_conf.seq_tfmr_num_heads,
                    dim_feedforward=tfmr_in,
                    batch_first=True,
                    dropout=self._ipa_conf.transformer_dropout,
                    norm_first=False
                )
                self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                    tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            elif self._ipa_conf.seq_tfmr_attention == "flash_attention":
                self.trunk[f'seq_tfmr_{b}'] = nn.ModuleList()
                for _ in range(self._ipa_conf.seq_tfmr_num_layers):
                    self.trunk[f'seq_tfmr_{b}'].append(Attention(c_q=tfmr_in,c_k=tfmr_in,c_v=tfmr_in,c_hidden=int(tfmr_in/self._ipa_conf.seq_tfmr_num_heads),no_heads=self._ipa_conf.seq_tfmr_num_heads,gating=False))

            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block
                if self._ipa_conf.local_triangle_attention_new.enable:
                    # use local edge triangle attention for better performance
                    self.trunk[f'edge_transition_{b}'] = local_triangle_attention.LocalTriangleAttentionNew(**self._ipa_conf.local_triangle_attention_new)
                else:
                    # use simple transition layer
                    edge_in = self.structure_model_cfg.edge_embed_size
                    self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(node_embed_size=self._ipa_conf.c_s, edge_embed_in=edge_in, edge_embed_out=self.structure_model_cfg.edge_embed_size,)
            
            """
            if b == self._ipa_conf.num_blocks - 1 and self.structure_model_cfg.auxiliary_heads.enable:
                self.auxiliary_heads = heads.AuxiliaryHeads(self.structure_model_cfg.auxiliary_heads)
            """
            

        # self.torsion_pred = ipa_pytorch.TorsionAngles(self._ipa_conf.c_s, 7)

    def forward(self, input_features):
        # Extract input features
        so3_t = input_features['so3_t']
        r3_t = input_features['r3_t']
        aatype = input_features['aatype']
        node_mask = input_features['residue_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        residue_index = input_features['residue_index']  
        chain_index = input_features['chain_index']
        scaffold_mask = input_features['scaffold_mask']
        motif_mask = input_features['motif_mask']
        interface_mask = torch.zeros_like(scaffold_mask)
        z_all = input_features['z_all']
        pred_contact_map = input_features['pred_contact_map']
        fixed_structure_mask = input_features['fixed_structure_mask']
        all_diffuse_mask = torch.ones_like(scaffold_mask) * node_mask

        trans_t = input_features['trans_t']
        rotmats_t = input_features['rotmats_t']
        trans_1 = input_features['trans_1']
        rotmats_1 = input_features['rotmats_1']

        # self-conditioning
        if 'trans_sc' not in input_features:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_features['trans_sc']
        if 'rotmats_sc' not in input_features:
            rotmats_sc = torch.zeros_like(rotmats_t)
        else:
            rotmats_sc = input_features['rotmats_sc']


        # Initialize embeddings
        init_node_embed = self.node_feature_net(
            so3_t, r3_t, residue_index, node_mask, chain_index,
            scaffold_mask, motif_mask, aatype, interface_mask, z_all,
        ) * node_mask[..., None]
        init_edge_embed = self.edge_feature_net(
            so3_t, init_node_embed, edge_mask, residue_index, chain_index, node_mask, 
            trans_1, rotmats_1, trans_t, rotmats_t, trans_sc, rotmats_sc, 
            scaffold_mask, motif_mask, fixed_structure_mask, pred_contact_map,
        ) * edge_mask[..., None]

        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]

        # pair transform net
        if self.structure_model_cfg.pair_transform_net.enable:
            #node_embed, edge_embed = self.pair_transform_net(init_node_embed, init_edge_embed)
            edge_embed = self.pair_transform_net(edge_embed, edge_mask)

        # Initialize rigids and apply scaling, masking
        # init_rigis = du.create_rigid(rotmats_t, trans_t)
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)

        # Main trunk processing
        for block_num in range(self._ipa_conf.num_blocks):
            # Apply Invariant Point Attention (IPA)
            ipa_embed = self.trunk[f'ipa_{block_num}'](node_embed, edge_embed, curr_rigids, node_mask, pred_contact_map=pred_contact_map, t=r3_t)
            ipa_embed *= node_mask[..., None]

            # Apply Layer Normalization and update node embedding
            node_embed = self.trunk[f'ipa_ln_{block_num}'](node_embed + ipa_embed)

            # Skip embedding
            if self._ipa_conf.c_skip_embedding.enable:
                # Apply Sequence Transformer
                seq_tfmr_in = torch.cat([node_embed, self.trunk[f'skip_embed_{block_num}'](init_node_embed)], dim=-1)
            else:
                seq_tfmr_in = node_embed    
            
            # Use pytorch or flash_attention
            if self._ipa_conf.seq_tfmr_attention == "pytorch":
                seq_tfmr_out = self.trunk[f'seq_tfmr_{block_num}'](seq_tfmr_in, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            elif self._ipa_conf.seq_tfmr_attention == "flash_attention":
                seq_tfmr_out = seq_tfmr_in
                for tfmr in self.trunk[f'seq_tfmr_{block_num}']:
                    seq_tfmr_out = tfmr(seq_tfmr_out, seq_tfmr_out, use_flash=True, flash_mask = node_mask)
            
            # Post-process Transformer output and update node embedding
            node_embed = node_embed + self.trunk[f'post_tfmr_{block_num}'](seq_tfmr_out)

            # Apply node transition and masking
            node_embed = self.trunk[f'node_transition_{block_num}'](node_embed)
            node_embed *= node_mask[..., None]

            # Update backbone
            rigid_update = self.trunk[f'bb_update_{block_num}'](node_embed)
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, all_diffuse_mask[..., None])

            # Update edge embedding if not the last block
            if block_num < self._ipa_conf.num_blocks - 1:
                curr_unscale_rigids = self.rigids_nm_to_ang(curr_rigids)
                if self._ipa_conf.local_triangle_attention_new.enable:
                    edge_embed = self.trunk[f'edge_transition_{block_num}'](node_embed, edge_embed, rigids=curr_unscale_rigids, edge_mask=edge_mask, contact_map=pred_contact_map, t=r3_t)
                else:
                    edge_embed = self.trunk[f'edge_transition_{block_num}'](node_embed, edge_embed)
                edge_embed *= edge_mask[...,None]
            
            """
            if block_num == self._ipa_conf.num_blocks - 1 and self.model_conf.auxiliary_heads.enable:
                aux_heads_output = self.auxiliary_heads(node_embed, edge_embed)
            """

        # Final processing
        # unnormalized_angles, pred_torsion = self.torsion_pred(node_embed)
        rigids_preds = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = rigids_preds.get_trans()
        pred_rotmats = rigids_preds.get_rots().get_rot_mats()

        # 'pred_torsion': pred_torsion,
        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            # 'auxiliary_heads': aux_heads_output if self.model_conf.auxiliary_heads.enable else None,
        }
import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Dict
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_contact_map(
        trans_1: torch.Tensor, # (B, N, 3),
        residue_mask: torch.Tensor, # (B, N)
        cutoff: float = 8.0
    ) -> torch.Tensor:
    ca_coords = trans_1  # (B, N, 3)
    diff = ca_coords.unsqueeze(2) - ca_coords.unsqueeze(1)  # (B, N, N, 3)
    distances = torch.norm(diff, dim=-1)  # (B, N, N)
    contact_map = (distances < cutoff).float()
    contact_map = contact_map * residue_mask.unsqueeze(1) * residue_mask.unsqueeze(2)
    return contact_map

def compute_vae_loss(
        pred_logits: torch.Tensor, # (B, N, N) : logits of contact
        true_contacts: torch.Tensor, # (B, N, N) : ground truth contact map
        mean: torch.Tensor, # (B, N, latent_dim) : mean of the latent distribution
        logvar: torch.Tensor, # (B, N, latent_dim) : log variance of the latent distribution
        motif_mask: torch.Tensor # (B, N) : motif mask
    ) -> Tuple[torch.Tensor, Dict]:
    
    # Due to the sparsity of the contact map, we use a weighted BCE loss.
    # we weight the positive and negative class seperately by its propensity.
    eps = 1e-8
    with torch.no_grad():
        alpha = true_contacts.mean().clamp(min=eps, max=1.0-eps)
        pos_weight = 1.0 / alpha
        neg_weight = 1.0 / (1.0 - alpha)
    
    weights = torch.where(true_contacts > 0.5, pos_weight, neg_weight)
    recon_map_loss = F.binary_cross_entropy_with_logits(
        pred_logits,     
        true_contacts,   
        weight=weights,  
        reduction='none'
    ) # (B, N, N)

    # Outer motif mask
    outer_motif_mask = motif_mask.unsqueeze(1) * motif_mask.unsqueeze(2)  # (B, N, N)
    recon_loss = (recon_map_loss * outer_motif_mask).sum() / (outer_motif_mask.sum() + eps)

    # KL divergence between q(z|x) and p(z) (motif residues only)
    # KL[q(z|x)||p(z)] = -0.5 * Σ(1 + logσ^2 - μ^2 - σ^2)
    kl_map = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())  # (B, N)
    kl_map = kl_map * motif_mask.unsqueeze(-1)
    kl_loss = kl_map.sum() / (motif_mask.sum() + eps)

    return recon_loss, kl_loss

def compute_vae_auc(
        pred_contact_map: torch.Tensor, # (B, N, N) : probability of contact
        true_contacts: torch.Tensor, # (B, N, N) : ground truth contact map
        motif_mask: torch.Tensor # (B, N) : motif mask
    ) -> Tuple[float, float]:
    
    mask_2d = motif_mask.unsqueeze(1) * motif_mask.unsqueeze(2)
    mask_2d = mask_2d.bool()
    valid_pred = pred_contact_map[mask_2d]
    valid_true = true_contacts[mask_2d]
    
    true_labels = valid_true.detach().cpu().numpy()
    pred_probs = valid_pred.detach().cpu().numpy()
    roc_auc = roc_auc_score(true_labels, pred_probs)
    pr_auc = average_precision_score(true_labels, pred_probs)
    
    return roc_auc, pr_auc

def calculate_kl_beta(current_epoch, total_annealing_epochs, beta_max=1.0, warmup_epochs=0):
    if current_epoch < warmup_epochs:
        return 0.0
    else:
        annealing_epoch = current_epoch - warmup_epochs
        if total_annealing_epochs <= 0:
             return beta_max
        anneal_progress = min(1.0, annealing_epoch / total_annealing_epochs)
        beta = beta_max * anneal_progress
        return beta

def calculate_kl_beta_cyclical(current_epoch, cycle_epoch_length, beta_max=1.0, warmup_epochs=0):
    if current_epoch < warmup_epochs:
        return 0.0
    else:
        effective_step = current_epoch - warmup_epochs
        step_in_cycle = effective_step % cycle_epoch_length
        if cycle_epoch_length <= 0:
            return beta_max 
        beta = beta_max * (step_in_cycle / cycle_epoch_length)
        beta = min(beta, beta_max)
        return beta

#################################################
# From LigandMPNN
#################################################

# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = torch.nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = torch.nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(torch.nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = torch.nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E

class ProteinmotifFeatures(torch.nn.Module):
    # From LigandMPNN proteinfeatures
    # https://github.com/dauparas/LigandMPNN/blob/main/model_utils.py#L1332
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=48,
        augment_eps=0.0,
    ):
        """Extract protein features"""
        super(ProteinmotifFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.edge_embedding = torch.nn.Linear(
            num_positional_embeddings + num_rbf * 25, # 5 atom * 5 atom 
            edge_features,
            bias=False
        )
        self.norm_edges = torch.nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, NCaCO_pos, residue_index, chain_index, motif_mask):

        if self.augment_eps > 0:
            NCaCO_pos = NCaCO_pos + self.augment_eps * torch.randn_like(NCaCO_pos)

        # Calculate Cb
        b = NCaCO_pos[:, :, 1, :] - NCaCO_pos[:, :, 0, :]
        c = NCaCO_pos[:, :, 2, :] - NCaCO_pos[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + NCaCO_pos[:, :, 1, :]
        Cb = Cb * motif_mask.unsqueeze(-1)
        Ca = NCaCO_pos[:, :, 1, :]
        N = NCaCO_pos[:, :, 0, :]
        C = NCaCO_pos[:, :, 2, :]
        O = NCaCO_pos[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, motif_mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_index[:, :, None] - residue_index[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_index[:, :, None] - chain_index[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, E_idx
    

class ProteinMPNN_EncLayer(torch.nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(ProteinMPNN_EncLayer, self).__init__()
        self.scale = scale
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(num_hidden)
        self.norm2 = torch.nn.LayerNorm(num_hidden)
        self.norm3 = torch.nn.LayerNorm(num_hidden)

        self.W1 = torch.nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = torch.nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E

class ProteinMPNN_DecLayer(torch.nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(ProteinMPNN_DecLayer, self).__init__()
        self.scale = scale
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(num_hidden)
        self.norm2 = torch.nn.LayerNorm(num_hidden)

        self.W1 = torch.nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1) # [B, N, K, 3*hidden_dim])

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V
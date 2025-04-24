from torch import nn
from openfold.model.dropout import DropoutRowwise, DropoutColumnwise
from openfold.model.pair_transition import PairTransition
from openfold.model.triangular_attention import(
	TriangleAttentionStartingNode,
	TriangleAttentionEndingNode,
)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)


class PairTransformLayer(nn.Module):
	"""
	Pair Transform Network (Pair Transform Layer in Genie2)

	Adapted from Evoformer, this module utilizes a triangular multiplicative
	update layer and a triangular attention layer (if specified) to refine
	pair representation.
	"""

	def __init__(
		self,
		c_p,
		include_mul_update,
		include_tri_att,
		c_hidden_mul,
		c_hidden_tri_att,
		n_head_tri,
		tri_dropout,
		pair_transition_n,
	):
		"""
		Args:
			c_p:
				Dimension of paired residue-residue (pair) representation.
			include_mul_update:
				Flag on whether to use triangular multiplicative update layer.
			include_tri_att:
				Flag on whether to use triangular attention layer.
			c_hidden_mul:
				Number of hidden dimensions in triangular multiplicative update layer.
			c_hidden_tri_att:
				Number of hidden dimensions in triangular attention layer.
			n_head_tri:
				Number of heads in triangular attention layer.
			tri_dropout:
				Dropout rate.
			pair_transition_n:
				Number of pair transition layers.
		"""
		super(PairTransformLayer, self).__init__()
		
        # Layers for triangular multiplicative updates
		self.tri_mul_out = TriangleMultiplicationOutgoing(
			c_p,
			c_hidden_mul
		) if include_mul_update else None
		self.tri_mul_in = TriangleMultiplicationIncoming(
			c_p,
			c_hidden_mul
		) if include_mul_update else None

		# Layers for triangular attention
		self.tri_att_start = TriangleAttentionStartingNode(
			c_p,
			c_hidden_tri_att,
			n_head_tri
		) if include_tri_att else None
		self.tri_att_end = TriangleAttentionEndingNode(
			c_p,
			c_hidden_tri_att,
			n_head_tri
		) if include_tri_att else None

		# Layer for pair transition
		self.pair_transition = PairTransition(
			c_p,
			pair_transition_n
		)

		# Layers for dropouts
		self.dropout_row_layer = DropoutRowwise(tri_dropout)
		self.dropout_col_layer = DropoutColumnwise(tri_dropout)
		
	def forward(self, inputs):
		"""
		Args:
			inputs:
				A tuple containing
					p:
						[B, N, N, c_p] pair representation
					pair_residue_mask:
						[B, N, N] pairwise residue mask.

		Returns:
			outputs:
				A tuple containing
					p:
						[B, N, N, c_p] updated pair representation
					pair_residue_mask:
						[B, N, N] pairwise residue mask.
		"""
		p, pair_residue_mask = inputs
		if self.tri_mul_out is not None:
			p = p + self.dropout_row_layer(self.tri_mul_out(p, pair_residue_mask))
			p = p + self.dropout_row_layer(self.tri_mul_in(p, pair_residue_mask))
		if self.tri_att_start is not None:
			p = p + self.dropout_row_layer(self.tri_att_start(p, pair_residue_mask))
			p = p + self.dropout_col_layer(self.tri_att_end(p, pair_residue_mask))
		p = p + self.pair_transition(p, pair_residue_mask)
		p = p * pair_residue_mask.unsqueeze(-1)
		outputs = (p, pair_residue_mask)
		return outputs

class PairTransformNet(nn.Module):
	"""
	Pair Transform Network.

	Adapted from Evoformer, this module utilizes multiple pair transform
	layers to refine pair representations before using them in the
	structure module.
	"""

	def __init__(
		self,
		c_p,
		n_pair_transform_layer,
		include_mul_update,
		include_tri_att,
		c_hidden_mul,
		c_hidden_tri_att,
		n_head_tri,
		tri_dropout,
		pair_transition_n,
		**kwargs,
	):
		"""
		Args:
			c_p:
				Dimension of paired residue-residue (pair) representation.
			n_pair_transform_layer:
				Number of pair transform layers.
			include_mul_update:
				Flag on whether to use triangular multiplicative update layer.
			include_tri_att:
				Flag on whether to use triangular attention layer.
			c_hidden_mul:
				Number of hidden dimensions in triangular multiplicative update layer.
			c_hidden_tri_att:
				Number of hidden dimensions in triangular attention layer.
			n_head_tri:
				Number of heads in triangular attention layer.
			tri_dropout:
				Dropout rate.
			pair_transition_n:
				Number of pair transition layers.
		"""
		super(PairTransformNet, self).__init__()

		# Create pair transform layers
		layers = [
			PairTransformLayer(
				c_p,
				include_mul_update,
				include_tri_att,
				c_hidden_mul,
				c_hidden_tri_att,
				n_head_tri,
				tri_dropout,
				pair_transition_n
			)
			for _ in range(n_pair_transform_layer)
		]

		# Create model
		self.net = nn.Sequential(*layers)

	def forward(self, p, pair_residue_mask):
		"""
		Args:
            p: pair representation [B,N,N,c_p]
			pair_residue_mask: pairwise residue mask [B,N,N]
			
		"""
		# Update pair representations
		# Shape: [B, N, N, c_p]
		p, _ = self.net((p, pair_residue_mask))

		return p
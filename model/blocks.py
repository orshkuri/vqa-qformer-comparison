from model.functionals import multi_head_attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


#################################################
# Multi Head Self Attention Layer
#################################################

class MHSA(nn.Module):
    def __init__(self, dim, num_heads):
        """Creates a Multi Head Self Attention layer.

        Args:
          dim (int): The input and output dimension (in this implementation Dy=Dq=Dk=Dv=Dx)
          num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # BEGIN SOLUTION
        self.lin_qkv = nn.Linear(dim, dim * 3)  # use this variable name
        self.lin = nn.Linear(dim, dim)  # use this variable name
        # END SOLUTION

    def forward(self, x, attention_mask=None):
        """Computes the `MHSA` of the input `x`.

        Args:
          x (torch.Tensor): The input tensor.
            Has shape `(batch_size, sequence_size, dim)`.

          attention_mask (Tensor, optional): A mask to indicate where attention should not be applied
                                                (batch_size, sequence_size, sequence_size).

        Returns:
          y (torch.Tensor): The output tensor.
            Has shape `(batch_size, sequence_size, dim)`.
        """

        # get dims
        b, s, d = x.shape

        # perform linear projections
        qkv = self.lin_qkv(x)  # b, s, d * 3

        # get projections (each of shape b, s, h, d//h)
        qkv = qkv.reshape(b, s, 3, self.num_heads, -1)
        qkv = qkv.permute(0, 3, 2, 1, 4)  # b, h, 3, s, d/h
        q, k, v = torch.unbind(qkv, dim=2)  # b, h, s, d/h

        y = multi_head_attention(q, k, v, attention_mask)  # b, s, d
        y = self.lin(y)
        return y


#################################################
# Multi Head Cross Attention Layer
#################################################


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        """
        Initializes the Multi-Head Cross Attention layer.
        """
        super().__init__()

        assert dim % num_heads == 0, f"Dimension {dim} must be divisible by number of heads {num_heads}"

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        # Linear transformations for Q, K, V
        self.lin_q = nn.Linear(dim, dim)
        self.lin_k = nn.Linear(dim, dim)
        self.lin_v = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        # Layer norm for better stability
        self.layer_norm = nn.LayerNorm(dim)

        # Dropout for regularization
        self.attn_dropout = nn.Dropout(0.3)
        self.proj_dropout = nn.Dropout(0.3)

    def forward(self, q, k, v, attention_mask=None):
        """
        Computes the Multi-Head Cross Attention.
        """
        batch_size, t_q, d_k = q.shape
        _, t_kv, _ = k.shape

        # Apply layer normalization for stability
        q = self.layer_norm(q)

        # Apply linear transformations first
        q = self.lin_q(q)
        k = self.lin_k(k)
        v = self.lin_v(v)

        # Reshape to multi-head format
        q = q.reshape(batch_size, t_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, t_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, t_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(attention_mask == 1, float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Transpose and reshape back
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, t_q, self.dim)

        # Apply output projection and dropout
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)

        return attn_output


#################################################
# Feed Forward
#################################################


class FeedForward(nn.Module):
    """
    Feed Forward Network with two linear layers and a GELU activation.
    Used in transformer blocks for point-wise transformations.
    """

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Apply feed forward transformation.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, dim)
        """
        return self.net(x)


class CrossModalTransformerLayer(nn.Module):
    """
    A single transformer layer for cross-modal processing.

    Includes cross-attention between queries and image features,
    self-attention on the concatenated queries and text embeddings,
    and a feed-forward network with layer normalization and residual connections.
    """

    def __init__(self, dim, num_heads, ffn_expansion_factor=4, dropout=0.1):
        super().__init__()

        # Cross-attention components
        self.norm_q_cross = nn.LayerNorm(dim)
        self.norm_kv_cross = nn.LayerNorm(dim)
        self.mhca = MultiHeadCrossAttention(dim, num_heads)
        self.dropout_cross = nn.Dropout(dropout)

        # Self-attention components
        self.norm_self = nn.LayerNorm(dim)
        self.mhsa = MHSA(dim, num_heads)
        self.dropout_self = nn.Dropout(dropout)

        # Feed-forward components
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ffn_expansion_factor, dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, queries, image_features, text_embeddings, attention_mask=None):
        """
        Process one layer of cross-modal transformer operations.

        Args:
            queries (Tensor): Query tensor of shape (batch_size, num_queries, dim)
            image_features (Tensor): Image features of shape (batch_size, num_patches, dim)
            text_embeddings (Tensor): Text embeddings of shape (batch_size, text_len, dim)
            attention_mask (Tensor, optional): Mask for self-attention: (batch_size, seq_len + text_len, seq_len + text_len)

        Returns:
            tuple: Updated queries and text_embeddings
        """
        # Cross-attention between queries and image features (with pre-normalization)
        q_norm = self.norm_q_cross(queries)
        kv_norm = self.norm_kv_cross(image_features)

        # Apply cross-attention (with residual connection)
        queries = queries + self.dropout_cross(self.mhca(q=q_norm, k=kv_norm, v=kv_norm))

        # Concatenate queries and text embeddings for self-attention
        combined = torch.cat([queries, text_embeddings], dim=1)

        # Apply self-attention (with pre-normalization and residual connection)
        combined_norm = self.norm_self(combined)

        # Apply self-attention with the combined mask
        combined = combined + self.dropout_self(self.mhsa(combined_norm, attention_mask))

        # Apply feed-forward network (with pre-normalization and residual connection)
        combined_norm = self.norm_ff(combined)
        combined = combined + self.dropout_ff(self.ff(combined_norm))

        # Split the combined tensor back into queries and text embeddings
        updated_queries, updated_text = torch.split(combined, [queries.size(1), text_embeddings.size(1)], dim=1)

        return updated_queries, updated_text


#################################################
# Cross Modal Transformer
#################################################


class CrossModalTransformer(nn.Module):
    """
    Cross Modal Transformer that processes information from multiple modalities.

    This transformer performs cross-attention between query embeddings and image features,
    followed by self-attention with text embeddings to integrate information across modalities.
    The model consists of multiple stacked transformer layers, each containing cross-attention,
    self-attention, and feed-forward components with normalization and residual connections.
    """

    def __init__(self, dim, num_heads, num_layers, ffn_expansion_factor=4, dropout=0.1):
        """
        Initializes the Cross Modal Transformer.

        Args:
            dim (int): The dimensionality of the input (also used for the output dimension).
            num_heads (int): The number of attention heads to split the dimensionality into.
            num_layers (int): The number of transformer layers to stack.
            ffn_expansion_factor (int, optional): Expansion factor for the feed-forward network.
                                                 Default: 4
            dropout (float, optional): Dropout probability. Default: 0.1
        """
        super().__init__()
        self.num_layers = num_layers

        # Create a stack of transformer layers
        self.layers = nn.ModuleList([
            CrossModalTransformerLayer(dim, num_heads, ffn_expansion_factor, dropout)
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize the weights of the model.

        For linear layers, uses truncated normal initialization.
        For layer normalization, uses ones for weight and zeros for bias.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, image_features, queries, text_embeddings, attention_mask=None):
        """
        Process multiple modalities through the Cross-Modal Transformer.

        Args:
            image_features (Tensor): Visual features extracted from an image,
                                    shape (batch_size, num_patches, dim)
            queries (Tensor): Query embeddings, shape (batch_size, num_queries, dim)
            text_embeddings (Tensor): Text embeddings, shape (batch_size, text_len, dim)
            attention_mask (Tensor, optional): Attention mask for self-attention,
                                             shape (batch_size, seq_len, seq_len)

        Returns:
            tuple: Final processed (queries, text_embeddings) after passing through all layers
        """
        # Process through each transformer layer
        for layer in self.layers:
            queries, text_embeddings = layer(
                queries=queries,
                image_features=image_features,
                text_embeddings=text_embeddings,
                attention_mask=attention_mask
            )

        # Apply final normalization to the queries (often the main output of interest)
        queries = self.final_layer_norm(queries)

        return queries, text_embeddings

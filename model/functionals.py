import torch.nn as nn
from torch import transpose, matmul, softmax


#################################################
# Multi Head Attention
#################################################


def multi_head_attention(q, k, v, attention_mask=None):
    """A differentiable multi head attention function.

  Args:
    q (torch.Tensor): The query embedding.
      Has shape `(batch_size, num_heads, sequence_size, head_emb_dim)`.
    k (torch.Tensor): The key embedding.
      Has shape `(batch_size, num_heads, sequence_size, head_emb_dim)`.
    v (torch.Tensor): The value embedding.
      Has shape `(batch_size, num_heads, sequence_size, head_emb_dim)`.

    attention_mask (Tensor, optional): A mask to indicate where attention should not be applied
                                                (batch_size, seq_len + text_len, seq_len + text_len)

  Returns:
    y (torch.Tensor): The multi head attention output.
      Has shape `(batch_size, sequence_size, num_heads * head_emb_dim)`.
  """
    # BEGIN SOLUTION
    b, h, s, e = q.shape
    # bhse
    k_transposed = transpose(k, -2, -1)  # bhes
    attention_mat = matmul(q, k_transposed) / (e ** 0.5)  # bhsc

    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(1)  # (b, 1, t_q, t_kv)  -> Add an extra dimension for heads.
        attention_mask = attention_mask.expand(-1, h, -1,
                                               -1)  # (b, h, t_q, t_kv) -> Expand to match the number of heads.

        # Apply the attention mask: Mask positions with value 1
        attention_mat = attention_mat.masked_fill(attention_mask == 1, float('-inf'))  # (b, h, t_q, t_kv)

    attention_mat = softmax(attention_mat, dim=-1)

    # bhsc, bhce -> bhse
    y = matmul(attention_mat, v)
    # concat heads
    y = y.permute(0, 2, 1, 3).reshape(b, s, -1)
    # END SOLUTION
    return y

import torch
import torch.nn as nn
import torch.nn.functional as F


# The provided cross_attention function
def cross_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == 1, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


# Your MultiHeadCrossAttention class
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.dim = dim

        self.lin_q = nn.Linear(dim, dim)
        self.lin_k = nn.Linear(dim, dim)
        self.lin_v = nn.Linear(dim, dim)

    def forward(self, q, k, v, attention_mask=None):

        """
        q: b, t, d
        k, v: b, s, d
        attention_mask: b, t, s
        """
        b, t_q, d_k = q.shape
        b, t_kv, _ = k.shape

        q = q.reshape(b, t_q, self.num_heads, -1).transpose(1, 2)  # b, h, t, d/h
        k = k.reshape(b, t_kv, self.num_heads, -1).transpose(1, 2)  # b, h, s, d/h
        v = v.reshape(b, t_kv, self.num_heads, -1).transpose(1, 2)  # b, h, s, d/h

        #  b h t d/h, b, h, d/h, s -> b, h, t, s
        scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)  # b, h, t, s
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)  # b, 1, t, s
            attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)  # b, h, t, s

            scores = scores.masked_fill(attention_mask == 1, float('-inf'))
        attent_weights = torch.softmax(scores, dim=-1)

        # b, h, t, s  ,  b, h, s, d/h  ->  b, h, t, d/h -> b, t, h, d/h -> b, t, d
        attn_output = (attent_weights @ v).transpose(1, 2).reshape(b, t_q, -1)

        return attn_output


# Test function to compare both models
def test_attention_models():
    # Hyperparameters
    batch_size = 2
    target_len = 5  # Length of the query (target sequence)
    source_len = 6  # Length of the key/value (source sequence)
    dim = 8  # Dimension of Q, K, V
    num_heads = 1  # Number of attention heads

    # Sample input data
    q = torch.randn(batch_size, target_len, dim)
    k = torch.randn(batch_size, source_len, dim)
    v = torch.randn(batch_size, source_len, dim)

    # Mask (optional)
    mask = torch.randint(0, 2,
                         (batch_size, target_len, source_len))  # 1 indicates no attention, 0 means attention allowed
    # mask = None

    # Initialize models
    model = MultiHeadCrossAttention(dim, num_heads)

    # Get the output from both functions
    cross_attention_output, _ = cross_attention(q, k, v, mask)
    mha_output = model(q, k, v, mask)

    # Check if the outputs are close enough (since MHA will introduce some randomness in values)
    assert torch.allclose(cross_attention_output, mha_output, atol=1e-6), "Outputs are not the same!"

    print("Both models produce the same output.")


# Run the test
if __name__ == "__main__":
    test_attention_models()

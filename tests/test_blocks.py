from model.blocks import MultiHeadCrossAttention
import pytest
import torch


class TestMultiHeadCrossAttention:
    @pytest.fixture
    def mhca(self):
        """Fixture to create a MultiHeadCrossAttention layer."""
        dim = 64
        num_heads = 8
        return MultiHeadCrossAttention(dim, num_heads)

    def test_output_shape(self, mhca):
        """
        Test that the output shape matches the input query shape.
        """
        batch_size = 2
        target_len = 10
        source_len = 15
        dim = 64
        num_heads = 8

        # Create random input tensors
        q = torch.randn(batch_size, target_len, dim)
        k = torch.randn(batch_size, source_len, dim)
        v = torch.randn(batch_size, source_len, dim)

        # Compute output
        output = mhca(q, k, v)

        # Check output shape
        assert output.shape == (batch_size, target_len, dim), \
            f"Output shape {output.shape} does not match expected {(batch_size, target_len, dim)}"

    def test_attention_mask(self, mhca):
        """
        Test the attention mask functionality.
        """
        batch_size = 2
        target_len = 10
        source_len = 15
        dim = 64

        # Create random input tensors
        q = torch.randn(batch_size, target_len, dim)
        k = torch.randn(batch_size, source_len, dim)
        v = torch.randn(batch_size, source_len, dim)

        # Create a mask that blocks some attention
        attention_mask = torch.zeros(batch_size, 1, 1, source_len, dtype=torch.bool)
        attention_mask[:, :, :, -5:] = 1  # Mask last 5 tokens

        # Compute output with mask
        output_masked = mhca(q, k, v, attention_mask)

        # Verify that the output is different from unmasked output
        output_unmasked = mhca(q, k, v)

        # Check that at least some values are different
        assert not torch.allclose(output_masked, output_unmasked), \
            "Masked output should be different from unmasked output"

    def test_conservation_of_information(self, mhca):
        """
        Test that the attention mechanism conserves information.

        The weighted sum of values should preserve the overall magnitude
        of the input value tensor.
        """
        batch_size = 2
        target_len = 10
        source_len = 15
        dim = 64

        # Create random input tensors
        q = torch.randn(batch_size, target_len, dim)
        k = torch.randn(batch_size, source_len, dim)
        v = torch.randn(batch_size, source_len, dim)

        # Compute output
        output = mhca(q, k, v)

        # Check that the output has similar magnitude to input
        assert torch.allclose(
            output.mean(),
            v.mean(),
            rtol=1e-2,
            atol=1e-2
        ), "Output mean should be close to input value mean"

    def test_multi_head_independence(self, mhca):
        """
        Test that different heads can attend to different parts of the input.

        This is done by checking that the attention weights across heads are not identical.
        """
        batch_size = 2
        target_len = 10
        source_len = 15
        dim = 64

        # Create random input tensors
        q = torch.randn(batch_size, target_len, dim)
        k = torch.randn(batch_size, source_len, dim)
        v = torch.randn(batch_size, source_len, dim)

        # Manually compute attention weights to check multi-head behavior
        def compute_attention_weights(q, k):
            # Reshape for multi-head
            b, t_q, _ = q.shape
            b, t_kv, _ = k.shape

            # Project and reshape
            q_heads = mhca.lin_q(q).reshape(b, t_q, mhca.num_heads, -1).transpose(1, 2)
            k_heads = mhca.lin_k(k).reshape(b, t_kv, mhca.num_heads, -1).transpose(1, 2)

            # Compute scores
            scores = (q_heads @ k_heads.transpose(-2, -1)) * (mhca.dim ** -0.5)

            # Compute attention weights
            return torch.softmax(scores, dim=-1)

        # Compute attention weights
        attention_weights = compute_attention_weights(q, k)

        # Check that attention weights are not identical across all heads
        for head in range(1, mhca.num_heads):
            assert not torch.allclose(
                attention_weights[:, 0],
                attention_weights[:, head],
                rtol=1e-4
            ), f"Attention weights for head 0 and head {head} should not be identical"

    def test_deterministic_output(self, mhca):
        """
        Test that the output is deterministic for the same inputs.
        """
        batch_size = 2
        target_len = 10
        source_len = 15
        dim = 64

        # Create random input tensors
        q = torch.randn(batch_size, target_len, dim)
        k = torch.randn(batch_size, source_len, dim)
        v = torch.randn(batch_size, source_len, dim)

        # Compute output twice
        output1 = mhca(q, k, v)
        output2 = mhca(q, k, v)

        # Check that outputs are identical
        assert torch.allclose(output1, output2), \
            "Multiple calls with same inputs should produce identical outputs"


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
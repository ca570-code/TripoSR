import pytest


torch = pytest.importorskip("torch")

from tsr.models.transformer.attention import Attention


def test_prepare_attention_mask_pads_to_target_length():
    attention = Attention(query_dim=4, heads=2, dim_head=2)
    mask = torch.tensor([[[-10000.0, 0.0, -10000.0]]])

    prepared = attention.prepare_attention_mask(mask, target_length=5, batch_size=1)

    assert prepared.shape == (2, 1, 5)
    torch.testing.assert_close(prepared[0, :, :3], mask[0])
    torch.testing.assert_close(prepared[0, :, 3:], torch.zeros(1, 2))
    torch.testing.assert_close(prepared[1], prepared[0])


def test_prepare_attention_mask_rejects_overlong_mask():
    attention = Attention(query_dim=4, heads=2, dim_head=2)
    mask = torch.zeros(1, 1, 6)

    with pytest.raises(ValueError, match="exceeds target length"):
        attention.prepare_attention_mask(mask, target_length=5, batch_size=1)

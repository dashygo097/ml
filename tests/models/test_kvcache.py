import unittest

import torch

import ml


class TestKVCache(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_kvcache(self):
        torch.manual_seed(0)
        attn = ml.MulHeadAttn(128, 8)
        x = torch.randn(1, 10, 128)
        new_x = torch.randn(1, 1, 128)
        records = ml.AttnInfraRecord(input_logits=x)
        new_records = ml.AttnInfraRecord(input_logits=torch.cat([x, new_x], dim=1))
        records = attn.prompt(records)
        records.input_logits = torch.cat([x, new_x], dim=1)
        records = attn.infer(records, use_cache=True)
        new_records = attn.prompt(new_records)
        self.assertTrue(
            torch.allclose(records.output_logits, new_records.output_logits, atol=1e-3)
        )
        print("[SUCCESS] test_kvcache passed!")


if __name__ == "__main__":
    unittest.main()

import time
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
        print("[SUCCESS] test_kvcache passed!\n")

    def test_kvcache_speed(self):
        torch.manual_seed(0)
        attn = ml.MulHeadAttn(128, 8)
        x = torch.randn(1, 512, 128)
        new_x = torch.randn(1, 1, 128)
        records = ml.AttnInfraRecord(input_logits=x)
        new_records = ml.AttnInfraRecord(input_logits=torch.cat([x, new_x], dim=1))

        records = attn.prompt(records)
        start_cached = time.time()
        records.input_logits = torch.cat([x, new_x], dim=1)
        records = attn.infer(records, use_cache=True)
        end_cached = time.time()
        print(f"With KV cache: {end_cached - start_cached:.6f} seconds")

        start = time.time()
        new_records = attn.prompt(new_records)
        end = time.time()
        print(f"Without KV cache: {end - start:.6f} seconds")

        print(f"[INFO] Speedup: {(end - start) / (end_cached - start_cached)}x")
        print("[SUCCESS] benchmark_kvcache completed!\n")


if __name__ == "__main__":
    unittest.main()

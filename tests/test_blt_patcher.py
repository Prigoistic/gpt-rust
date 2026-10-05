import torch

from blt.patcher import next_byte_entropy, patch_lengths, patch_start_ids


def reference_starts(entropies: list[float], threshold: float) -> list[int]:
    starts = [p for p in range(len(entropies)) if p < 2 or entropies[p - 1] > threshold]
    return [p for p in starts if p < len(entropies)]


def test_entropy_of_uniform_distribution_is_log_vocab():
    logits = torch.zeros(2, 3, 260)
    assert torch.allclose(next_byte_entropy(logits), torch.full((2, 3), 260.0).log(), atol=1e-5)


def test_entropy_of_a_peaked_distribution_is_near_zero():
    logits = torch.full((1, 1, 10), -50.0)
    logits[..., 3] = 50.0
    assert next_byte_entropy(logits).item() < 1e-6


def test_hand_checked_example():
    entropies = torch.tensor([9.0, 0.1, 2.0, 0.2, 3.0, 0.1])
    assert patch_start_ids(entropies, 1.0).tolist() == [0, 1, 3, 5]
    assert patch_lengths(entropies, 1.0).tolist() == [1, 2, 2, 1]


def test_last_entropy_is_ignored_because_it_predicts_past_the_end():
    entropies = torch.tensor([0.0, 0.0, 0.0, 9.0])
    assert patch_lengths(entropies, 1.0).tolist() == [1, 3]


def test_extreme_thresholds():
    entropies = torch.rand(10) * 3
    assert patch_lengths(entropies, 1e9).tolist() == [1, 9]
    assert patch_lengths(entropies, -1.0).tolist() == [1] * 10


def test_tiny_sequences():
    assert patch_lengths(torch.tensor([0.5]), 1.0).tolist() == [1]
    assert patch_lengths(torch.tensor([0.5, 0.5]), 1.0).tolist() == [1, 1]


def test_matches_loop_reference_and_lengths_sum_to_sequence_length():
    torch.manual_seed(0)
    for _ in range(200):
        n = int(torch.randint(1, 60, ()))
        entropies = torch.rand(n) * 3
        t = float(torch.rand(()) * 2)
        starts = patch_start_ids(entropies, t).tolist()
        assert starts == reference_starts(entropies.tolist(), t)
        lengths = patch_lengths(entropies, t)
        assert lengths.sum().item() == n and (lengths >= 1).all()

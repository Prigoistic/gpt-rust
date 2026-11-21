import pytest

from blt_baseline.comparison import paired_summary


def observation(seconds, ids=None):
    return {"elapsed_seconds": seconds, "raw_byte_count": 8,
            "generated_ids": [[69]] if ids is None else ids,
            "peak_allocated_bytes": 123, "fallback_rows": 1}


def test_paired_summary_uses_paired_ratios_and_keeps_absolute_rate():
    samples = [{"padded": observation(t), "fast": observation(t / 4)} for t in (4, 8, 12)]
    result = paired_summary(samples)
    assert result["fast"]["paired_speedup_median"] == 4
    assert result["fast"]["paired_speedup_bootstrap_95"] == pytest.approx([4, 4])
    assert result["fast"]["raw_bytes_per_second_per_gpu"] == 4
    assert result["fast"]["fallback_rows"] == 3
    assert result["fast"]["all_repeated_ids_equal"]


def test_changed_output_or_missing_pair_cannot_be_reported_as_speedup():
    with pytest.raises(ValueError, match="different IDs"):
        paired_summary([{"padded": observation(4), "fast": observation(1, [[70]])}])
    with pytest.raises(ValueError, match="same policies"):
        paired_summary([{"padded": observation(4), "fast": observation(1)}, {"padded": observation(4)}])

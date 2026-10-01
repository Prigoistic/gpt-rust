import torch

from blt.patching import (
    completed_patch,
    decoder_cross_mask,
    encoder_cross_mask,
    max_pool_patches,
    patch_ids_from_starts,
)

STARTS = torch.tensor([[1, 0, 0, 1, 0, 1]], dtype=torch.bool)


def test_patch_ids_from_starts():
    assert patch_ids_from_starts(STARTS).tolist() == [[0, 0, 0, 1, 1, 2]]


def test_max_pool_takes_the_max_within_each_patch():
    ids = patch_ids_from_starts(STARTS)
    x = torch.tensor([[[1.0], [5.0], [2.0], [3.0], [-1.0], [7.0]]])
    assert max_pool_patches(x, ids, 3).flatten().tolist() == [5.0, 3.0, 7.0]


def test_max_pool_empty_patch_is_zero():
    ids = torch.tensor([[0, 0, 2]])
    x = torch.tensor([[[4.0], [-3.0], [9.0]]])
    assert max_pool_patches(x, ids, 3).flatten().tolist() == [4.0, 0.0, 9.0]


def test_encoder_mask_selects_own_bytes_only():
    mask = encoder_cross_mask(patch_ids_from_starts(STARTS), 3)[0]
    assert mask.int().tolist() == [
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]


def test_completed_patch_is_the_latest_finished_one():
    ids = patch_ids_from_starts(STARTS)
    assert completed_patch(ids).tolist() == [[-1, -1, 0, 0, 1, 2]]


def test_decoder_mask_never_exposes_unfinished_patches():
    ids = patch_ids_from_starts(STARTS)
    mask = decoder_cross_mask(ids, 3)[0]
    assert mask.int().tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]

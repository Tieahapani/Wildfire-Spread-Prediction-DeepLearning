import pytest
import torch

from wfspread.constants import CH, IGNORE_INDEX, NDWS_CHANNELS
from wfspread.data.batch import check_batch


def _batch(n_tiles: int = 2, size: int = 8, num_channels: int = len(NDWS_CHANNELS)) -> dict:
    g = torch.Generator().manual_seed(0)
    return {
        "inputs": torch.randn(n_tiles, num_channels, size, size, generator=g),
        "prev_fire": torch.randint(-1, 2, (n_tiles, size, size), generator=g, dtype=torch.int8),
        "next_fire": torch.randint(-1, 2, (n_tiles, size, size), generator=g, dtype=torch.int8),
        "sample_id": torch.arange(n_tiles, dtype=torch.int64),
    }


def test_channel_order_is_fixed():
    assert len(NDWS_CHANNELS) == 13
    assert len(set(NDWS_CHANNELS)) == len(NDWS_CHANNELS)
    assert CH["wind_dir_sin"] == 1
    assert CH["prev_fire"] == 12


def test_valid_batch_passes():
    check_batch(_batch())


def test_crop_sized_batch_passes():
    check_batch(_batch(size=32))


def test_inference_batch_without_masks_passes():
    batch = _batch()
    del batch["prev_fire"], batch["next_fire"]
    check_batch(batch)


def test_dropped_channel_passes_with_matching_count():
    check_batch(_batch(num_channels=12), num_channels=12)


@pytest.mark.parametrize("key", ["inputs", "sample_id"])
def test_missing_required_key_fails(key):
    batch = _batch()
    del batch[key]
    with pytest.raises(ValueError, match=key):
        check_batch(batch)


def test_wrong_channel_count_fails():
    with pytest.raises(ValueError, match="channels"):
        check_batch(_batch(num_channels=12))


def test_non_finite_inputs_fail():
    batch = _batch()
    batch["inputs"][0, CH["wind_dir_sin"], 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        check_batch(batch)


def test_float64_inputs_fail():
    batch = _batch()
    batch["inputs"] = batch["inputs"].double()
    with pytest.raises(ValueError, match="float32"):
        check_batch(batch)


def test_int64_mask_fails():
    batch = _batch()
    batch["next_fire"] = batch["next_fire"].long()
    with pytest.raises(ValueError, match="next_fire"):
        check_batch(batch)


def test_mask_size_must_match_inputs():
    batch = _batch()
    batch["prev_fire"] = batch["prev_fire"][:, :4, :4].contiguous()
    with pytest.raises(ValueError, match="prev_fire"):
        check_batch(batch)


def test_unknown_mask_value_fails():
    batch = _batch()
    batch["next_fire"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="values"):
        check_batch(batch)


def test_ignore_only_mask_passes():
    batch = _batch()
    batch["next_fire"].fill_(IGNORE_INDEX)
    check_batch(batch)


def test_sample_id_length_must_match():
    batch = _batch()
    batch["sample_id"] = torch.arange(3, dtype=torch.int64)
    with pytest.raises(ValueError, match="sample_id"):
        check_batch(batch)

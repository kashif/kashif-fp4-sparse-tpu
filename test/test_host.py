import pytest

from software.fp4_tpu import (
    accumulate_scaled_partials,
    decode_result,
    load_activation,
    load_weight,
    run,
    sparse_weight_code,
    store,
)


def test_instruction_encoding_matches_rtl_isa():
    assert load_activation(2, 7, -128) == 0b10_0_10_111_10000000
    code = sparse_weight_code(1, 0xF)
    assert code == 0x1F
    assert load_weight(2, 3, code) == 0b10_1_10_011_00011111
    assert run(False) == 0x4000
    assert run(True) == 0x6000
    assert store(2, 1, True) == 0b11_1_10_01_000000000


@pytest.mark.parametrize("value", [-8192, -6144, -1, 0, 1, 6096, 6144, 8191])
def test_signed_result_round_trip(value):
    raw = value & 0x3FFF
    assert decode_result(raw & 0xFF, raw >> 8) == value


def test_scale_each_block_before_accumulation():
    partials = [120, -40]
    scales = [0.5, 8.0]
    correct = accumulate_scaled_partials(partials, scales,
                                         activation_scale=0.25,
                                         tensor_scale=2.0)
    reference = sum((scale / 2) * partial * 0.25 * 2.0
                    for partial, scale in zip(partials, scales))
    naive = (scales[0] / 2) * sum(partials) * 0.25 * 2.0
    assert correct == reference
    assert correct != naive


def test_host_validation_rejects_out_of_range_fields():
    with pytest.raises(ValueError):
        load_activation(3, 0, 0)
    with pytest.raises(ValueError):
        load_weight(0, 4, 0)
    with pytest.raises(ValueError):
        sparse_weight_code(2, 0)
    with pytest.raises(ValueError):
        decode_result(256, 0)

"""Dependency-free host protocol and dequantization helpers.

The transport itself is platform-specific: send each returned 16-bit word
LSB-first with SCLK <= project_clk/6, keep CS high for at least four project
clock cycles between frames, then read ``uo_out`` after STORE. Poll sticky
``ready`` after RUN; it clears when the next RUN is accepted.
"""

OP_RUN = 0b01
OP_LOAD = 0b10
OP_STORE = 0b11


def _check_range(name, value, limit):
    if not 0 <= value < limit:
        raise ValueError(f"{name} must be in [0, {limit - 1}], got {value}")


def load_activation(row, element, value):
    """Encode LOAD A for one signed INT8 activation."""
    _check_range("row", row, 3)
    _check_range("element", element, 8)
    if not -128 <= value <= 127:
        raise ValueError(f"activation must be INT8, got {value}")
    return (OP_LOAD << 14) | (row << 11) | (element << 8) | (value & 0xFF)


def sparse_weight_code(select, e2m1_nibble):
    """Pack one 1:2 weight as ``{select, E2M1}``."""
    _check_range("select", select, 2)
    _check_range("E2M1 nibble", e2m1_nibble, 16)
    return (select << 4) | e2m1_nibble


def load_weight(column, slot, code):
    """Encode LOAD B for one five-bit sparse/dense weight code."""
    _check_range("column", column, 3)
    _check_range("slot", slot, 4)
    _check_range("weight code", code, 32)
    return (OP_LOAD << 14) | (1 << 13) | (column << 11) | (slot << 8) | code


def run(dense=False):
    """Encode RUN; dense=False selects 1:2 sparse K=8 operation."""
    return (OP_RUN << 14) | (int(bool(dense)) << 13)


def store(row, column, high_byte=False):
    """Encode STORE for one result byte."""
    _check_range("row", row, 3)
    _check_range("column", column, 3)
    return ((OP_STORE << 14) | (int(bool(high_byte)) << 13) |
            (row << 11) | (column << 9))


def decode_result(low_byte, high_byte):
    """Reconstruct one signed 14-bit result from two STORE reads."""
    _check_range("low byte", low_byte, 256)
    _check_range("high byte", high_byte, 256)
    value = ((high_byte & 0x3F) << 8) | low_byte
    return value - 0x4000 if value & 0x2000 else value


def dequantize_partial(integer_partial, weight_scale,
                       activation_scale=1.0, tensor_scale=1.0):
    """Scale one block partial from the chip's x2 integer domain."""
    return (integer_partial * (weight_scale / 2) *
            activation_scale * tensor_scale)


def accumulate_scaled_partials(integer_partials, weight_scales,
                               activation_scale=1.0, tensor_scale=1.0):
    """Dequantize each block before cross-block accumulation.

    ``integer_partials[b]`` and ``weight_scales[b]`` must refer to the same
    NVFP4/MXFP4 scale block. Applying one scale after summing blocks is wrong
    when their scales differ.
    """
    if len(integer_partials) != len(weight_scales):
        raise ValueError("partials and scales must have the same length")
    return sum(dequantize_partial(p, s, activation_scale, tensor_scale)
               for p, s in zip(integer_partials, weight_scales))

# SPDX-FileCopyrightText: (c) 2026 Kashif
# SPDX-License-Identifier: Apache-2.0
#
# FP4 sparse mini-TPU tests.
#
# The golden model is an INDEPENDENT dense matrix multiply built from
# first principles (decode {select, e2m1} codes into a dense 8x3 matrix
# via the E2M1 value table, then plain C = A @ W) — it shares no
# structure with the RTL, so it cannot "pass artificially" by mirroring
# implementation quirks.

import os
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

GL_TEST = bool(os.environ.get("GATES") == "yes")

N = 3      # array is N x N
K = 8      # contraction depth (K/2 = 4 sparse pair slots)

OP_RUN   = 0b01
OP_LOAD  = 0b10
OP_STORE = 0b11

# SPI pin positions within ui_in
PIN_MOSI = 0
PIN_CS   = 1
PIN_SCLK = 2

# SCLK half-period in clk cycles (SCLK = clk/8, well under the clk/6 limit)
SCLK_HALF = 4

# E2M1 magnitude table, x2 integer domain: code 0..7 -> value
E2M1_MAG = [0, 1, 2, 3, 4, 6, 8, 12]


# ----------------------------------------------------------------------
# Instruction encoding (16 bits, sent LSB-first)
# ----------------------------------------------------------------------

def instr_load_a(row, elem, value):
    return (OP_LOAD << 14) | (0 << 13) | (row << 11) | (elem << 8) | (value & 0xFF)


def instr_load_b(col, slot, code):
    return (OP_LOAD << 14) | (1 << 13) | (col << 11) | (slot << 8) | (code & 0x1F)


def instr_run(dense=0):
    return (OP_RUN << 14) | (dense << 13)


def instr_store(row, col, byte_sel):
    return (OP_STORE << 14) | (byte_sel << 13) | (row << 11) | (col << 9)


# ----------------------------------------------------------------------
# Golden model
# ----------------------------------------------------------------------

def e2m1_decode(nibble):
    """4-bit E2M1 -> signed value in the x2 integer domain (-0 = 0)."""
    mag = E2M1_MAG[nibble & 0x7]
    return -mag if nibble & 0x8 else mag


def code_decode(code):
    """{select, e2m1[3:0]} -> (select, signed weight value)."""
    return (code >> 4) & 1, e2m1_decode(code & 0xF)


def decode_weights(wcodes):
    """12 sparse codes (wcodes[col][slot]) -> dense 8x3 signed matrix."""
    W = [[0] * N for _ in range(K)]
    for c in range(N):
        for j in range(K // 2):
            sel, val = code_decode(wcodes[c][j])
            W[2 * j + sel][c] = val
    return W


def golden_matmul(A, wcodes):
    """C = A (3x8 signed INT8) x dense(W) (8x3). Exact — max |C| = 6144."""
    W = decode_weights(wcodes)
    return [[sum(A[i][k] * W[k][c] for k in range(K)) for c in range(N)]
            for i in range(N)]


def golden_dense_e2m1(A, wcodes):
    """Dense E2M1 mode: each code's nibble is a weight for ONE step;
    the select bit is ignored and only even activation slots
    (elements 0, 2, 4, 6) participate (K=4). Exact — max |C| = 6144."""
    return [[sum(A[i][2 * j] * e2m1_decode(wcodes[c][j] & 0xF)
                 for j in range(K // 2))
             for c in range(N)] for i in range(N)]


# ----------------------------------------------------------------------
# SPI driver
# ----------------------------------------------------------------------

async def spi_send(dut, instr):
    """Bit-bang one 16-bit instruction, LSB-first, sampled on SCLK rising."""
    def drive(mosi, cs, sclk):
        dut.ui_in.value = (mosi << PIN_MOSI) | (cs << PIN_CS) | (sclk << PIN_SCLK)

    drive(0, 0, 0)
    await ClockCycles(dut.clk, SCLK_HALF)
    for i in range(16):
        bit = (instr >> i) & 1
        drive(bit, 0, 0)                      # setup MOSI while SCLK low
        await ClockCycles(dut.clk, SCLK_HALF)
        drive(bit, 0, 1)                      # rising edge samples the bit
        await ClockCycles(dut.clk, SCLK_HALF)
    drive(0, 1, 0)                            # CS high between instructions
    # Leave time for the clk-domain data_ready pulse and execution.
    # RUN now time-multiplexes one PE across all 9 outputs: 1 issue
    # cycle + 9 x (4 accumulate + 1 commit) = 46 cycles worst case;
    # this gap covers it with margin.
    await ClockCycles(dut.clk, 60)


async def spi_abort_after_15_bits(dut, instr):
    """Send an incomplete frame, deassert CS, then clock SCLK while idle.

    A partial frame must be discarded rather than decoded when the old
    bit-counter value is reset.  Fifteen bits are intentional: for RUN,
    the omitted bit 15 is zero, so the partial buffer otherwise looks like
    a complete RUN instruction.
    """
    def drive(mosi, cs, sclk):
        dut.ui_in.value = (mosi << PIN_MOSI) | (cs << PIN_CS) | (sclk << PIN_SCLK)

    drive(0, 0, 0)
    await ClockCycles(dut.clk, SCLK_HALF)
    for i in range(15):
        bit = (instr >> i) & 1
        drive(bit, 0, 0)
        await ClockCycles(dut.clk, SCLK_HALF)
        drive(bit, 0, 1)
        await ClockCycles(dut.clk, SCLK_HALF)

    # Abort, then provide an idle SCLK edge. The legacy wrap detector
    # incorrectly treated this reset edge as completion of the 15-bit word.
    drive(0, 1, 0)
    await ClockCycles(dut.clk, SCLK_HALF)
    drive(0, 1, 1)
    await ClockCycles(dut.clk, SCLK_HALF)
    drive(0, 1, 0)
    await ClockCycles(dut.clk, 60)


async def hw_reset(dut):
    dut.ena.value = 1
    dut.ui_in.value = 1 << PIN_CS   # CS idle high
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


# ----------------------------------------------------------------------
# High-level operations
# ----------------------------------------------------------------------

async def load_operands(dut, A, wcodes):
    for i in range(N):
        for k in range(K):
            await spi_send(dut, instr_load_a(i, k, A[i][k]))
    for c in range(N):
        for j in range(K // 2):
            await spi_send(dut, instr_load_b(c, j, wcodes[c][j]))


async def read_result(dut, row, col):
    await spi_send(dut, instr_store(row, col, 0))
    low = int(dut.uo_out.value)
    await spi_send(dut, instr_store(row, col, 1))
    high = int(dut.uo_out.value)
    val = ((high & 0x3F) << 8) | low        # 14-bit signed accumulator
    if val & 0x2000:
        val -= 0x4000
    return val


async def run_matmul(dut, A, wcodes, dense=0):
    await load_operands(dut, A, wcodes)
    await spi_send(dut, instr_run(dense))
    return [[await read_result(dut, i, c) for c in range(N)] for i in range(N)]


def check(dut, got, expected, label):
    for i in range(N):
        for c in range(N):
            assert got[i][c] == expected[i][c], (
                f"{label}: C[{i}][{c}] expected {expected[i][c]}, "
                f"got {got[i][c]} (full: got={got} expected={expected})")


def start_clock(dut):
    cocotb.start_soon(Clock(dut.clk, 200, unit="ns").start())


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

@cocotb.test()
async def test_known_matmul(dut):
    """Hand-checked matmul: distinct INT8 activations, mixed E2M1 weights."""
    start_clock(dut)
    await hw_reset(dut)

    A = [[100, -128, 3, 45, 0, -2, 77, -9],
         [0, 17, 22, -33, 1, 127, -128, 8],
         [-1, -100, 2, 66, -4, 3, 19, -45]]
    # wcodes[col][slot] = {select, e2m1}: sel=1 puts the value at k=2j+1
    wcodes = [[0x03, 0x15, 0x02, 0x1D],  # col 0: W[0]=1.5, W[3]=3, W[4]=1, W[7]=-3
              [0x1E, 0x04, 0x11, 0x06],  # col 1: W[1]=-4, W[2]=2, W[5]=0.5, W[6]=4
              [0x0F, 0x00, 0x17, 0x1A]]  # col 2: W[0]=-6, (zero), W[5]=6, W[7]=-1

    results = await run_matmul(dut, A, wcodes)
    dut._log.info(f"results: {results}")
    check(dut, results, golden_matmul(A, wcodes), "known")
    dut._log.info("known matmul PASSED")


@cocotb.test()
async def test_select_bit_semantics(dut):
    """The select bit must pick position k=2j (0) or k=2j+1 (1)."""
    start_clock(dut)
    await hw_reset(dut)

    # Activations distinct at every k so misrouting is visible
    A = [[10, 20, 30, 40, 50, 60, 70, 80],
         [70, -80, -70, -60, -50, -40, -30, -20],
         [-30, -20, -10, 15, 25, 35, 45, 55]]

    # Single weight: E2M1 code 5 (= value 6) in col 0 slot 1 (covers k=2,3)
    for sel in (0, 1):
        wcodes = [[0x00, (sel << 4) | 0x05, 0x00, 0x00],
                  [0x00, 0x00, 0x00, 0x00],
                  [0x00, 0x00, 0x00, 0x00]]
        results = await run_matmul(dut, A, wcodes)
        for i in range(N):
            expected = 6 * A[i][2 + sel]
            assert results[i][0] == expected, (
                f"sel={sel}: C[{i}][0] expected {expected}, got {results[i][0]}")
            assert results[i][1] == 0 and results[i][2] == 0
    dut._log.info("select-bit semantics PASSED")


@cocotb.test()
async def test_negative_zero(dut):
    """E2M1 -0 (nibble 0x8) must contribute exactly zero, with either
    select bit — it is NOT -8."""
    start_clock(dut)
    await hw_reset(dut)

    A = [[127, -128, 99, -99, 55, -55, 111, -111]] * 3
    wcodes = [[0x08, 0x18, 0x08, 0x18],   # -0 at every slot, both selects
              [0x08, 0x18, 0x08, 0x18],
              [0x02, 0x08, 0x18, 0x08]]   # one real weight: W[0]=1(2) in col 2

    results = await run_matmul(dut, A, wcodes)
    expected = golden_matmul(A, wcodes)
    check(dut, results, expected, "neg-zero")
    for i in range(N):
        assert results[i][0] == 0 and results[i][1] == 0
    dut._log.info("negative zero PASSED")


@cocotb.test()
async def test_not_degenerate(dut):
    """Two activation matrices with identical row sums must give
    different results — guards against the w*sum(acts) failure mode
    the earlier designs had."""
    start_clock(dut)
    await hw_reset(dut)

    A1 = [[10, 20, 30, 40, 50, 60, 70, 80]] * 3
    A2 = [[80, 70, 60, 50, 40, 30, 20, 10]] * 3   # same row sums, reversed
    wcodes = [[0x01, 0x02, 0x03, 0x04],     # W[0]=1, W[2]=2, W[4]=3, W[6]=4
              [0x11, 0x12, 0x13, 0x14],     # W[1]=1, W[3]=2, W[5]=3, W[7]=4
              [0x0C, 0x00, 0x00, 0x00]]     # W[0]=-4

    r1 = await run_matmul(dut, A1, wcodes)
    r2 = await run_matmul(dut, A2, wcodes)
    check(dut, r1, golden_matmul(A1, wcodes), "A1")
    check(dut, r2, golden_matmul(A2, wcodes), "A2")
    assert r1 != r2, ("equal-sum inputs gave identical outputs — "
                      "design has collapsed to w*sum(acts) again")
    dut._log.info("non-degeneracy PASSED")


@cocotb.test()
async def test_run_clears_accumulators(dut):
    """Each RUN starts from zero — results must not double on rerun."""
    start_clock(dut)
    await hw_reset(dut)

    A = [[11, 12, 21, 22, 31, 32, 41, 42],
         [41, 42, 51, 52, 61, 62, 71, 72],
         [-11, -22, -33, -44, -55, -66, -77, -88]]
    wcodes = [[0x07, 0x17, 0x07, 0x17],
              [0x16, 0x06, 0x16, 0x06],
              [0x05, 0x15, 0x05, 0x15]]

    expected = golden_matmul(A, wcodes)
    await load_operands(dut, A, wcodes)
    await spi_send(dut, instr_run())
    await spi_send(dut, instr_run())    # second RUN, same operands
    results = [[await read_result(dut, i, c) for c in range(N)]
               for i in range(N)]
    check(dut, results, expected, "rerun")
    dut._log.info("accumulator clear PASSED")


@cocotb.test()
async def test_spi_partial_frame_abort(dut):
    """Deasserting CS after 15 bits must discard the apparent RUN."""
    start_clock(dut)
    await hw_reset(dut)

    A = [[1] * K for _ in range(N)]
    # Four +6 weights per column would make every result visibly nonzero.
    wcodes = [[0x07] * (K // 2) for _ in range(N)]
    await load_operands(dut, A, wcodes)

    await spi_abort_after_15_bits(dut, instr_run())
    results = [[await read_result(dut, i, c) for c in range(N)]
               for i in range(N)]
    assert results == [[0] * N for _ in range(N)], (
        f"aborted 15-bit RUN executed unexpectedly: {results}")
    dut._log.info("partial SPI frame abort PASSED")


@cocotb.test()
async def test_dense_e2m1_mode(dut):
    """Dense E2M1 mode (RUN with dense=1): each code's nibble is a
    weight for ONE contraction step (K=4, half throughput) — dense
    INT8 x E2M1, the format NVFP4 pretraining/inference actually uses.
    Select bits are set to garbage to prove they are ignored, and odd
    activation slots hold garbage to prove they don't participate."""
    start_clock(dut)
    await hw_reset(dut)

    # Real activations at even elements 0,2,4,6; garbage at odd elements
    A = [[3, -88, -5, 77, 2, -18, 9, 66],
         [-7, 55, 6, -18, -1, 7, -128, -3],
         [4, -3, -128, 6, 127, -2, 31, 90]]

    # Full E2M1 range incl. -0; select bits (0x10) set at random
    wcodes = [[0x18, 0x07, 0x1F, 0x03],  # col 0: -0, 6, -6, 1.5
              [0x05, 0x1B, 0x00, 0x1E],  # col 1: 3, -1.5, 0, -4
              [0x0C, 0x02, 0x19, 0x08]]  # col 2: -2, 1, -0.5, -0

    results = await run_matmul(dut, A, wcodes, dense=1)
    dut._log.info(f"dense results: {results}")
    check(dut, results, golden_dense_e2m1(A, wcodes), "dense-e2m1")

    # Same operands in sparse mode must honor the select bits again
    # (mode is latched per RUN, not sticky)
    results_sparse = await run_matmul(dut, A, wcodes, dense=0)
    check(dut, results_sparse, golden_matmul(A, wcodes), "back-to-sparse")
    dut._log.info("dense E2M1 mode PASSED")


@cocotb.test()
async def test_random(dut):
    """Randomized full-coverage trials against the golden model,
    random mode each trial."""
    start_clock(dut)
    await hw_reset(dut)

    rng = random.Random(0x1247)
    trials = 3 if GL_TEST else 12

    for t in range(trials):
        A = [[rng.randint(-128, 127) for _ in range(K)] for _ in range(N)]
        wcodes = [[rng.randint(0, 31) for _ in range(K // 2)]
                  for _ in range(N)]
        dense = rng.randint(0, 1)
        results = await run_matmul(dut, A, wcodes, dense=dense)
        golden = golden_dense_e2m1(A, wcodes) if dense else golden_matmul(A, wcodes)
        check(dut, results, golden, f"trial {t} (dense={dense})")
        dut._log.info(f"trial {t} OK (dense={dense})")

    dut._log.info(f"random test PASSED ({trials} trials)")


# ----------------------------------------------------------------------
# NVFP4 four-over-six quantizer (arXiv:2512.02010), spec-derived
# ----------------------------------------------------------------------
# Host-side reference implementation of adaptive block scaling: each
# 16-element block is quantized twice — decode scale D = e4m3(amax/6)
# and D = e4m3(amax/4) — and the lower-MSE variant is kept.  Which D
# was chosen is invisible to the chip: it only ever sees E2M1 codes and
# returns exact integer partial sums, so 4/6 needs zero RTL support.
# (The paper's additional FP32 per-tensor scale is one more host-side
# multiply on the same exact sums; it is omitted here.)

BLOCK = 16                       # NVFP4 block size = 2 sparse RUNs

E2M1_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]   # code 0..7


def _round_nearest(x, table):
    """Nearest value in a sorted table, ties to the even-mantissa
    entry (even index — RTN as FP casts do). x must be >= 0."""
    best, best_err = None, None
    for idx, v in enumerate(table):
        err = abs(x - v)
        if (best_err is None or err < best_err
                or (err == best_err and idx % 2 == 0)):
            best, best_err = v, err
    return best


def e4m3_positives():
    """All positive finite E4M3 values: subnormals m/8 * 2^-6,
    normals (1+m/8) * 2^(e-7) for e=1..15, code (15,7) is NaN -> max 448."""
    vals = [m / 8.0 * 2.0 ** -6 for m in range(1, 8)]
    vals += [(1 + m / 8.0) * 2.0 ** (e - 7)
             for e in range(1, 16) for m in range(8) if not (e == 15 and m == 7)]
    return sorted(vals)


E4M3_POS = e4m3_positives()


def e4m3_quantize(x):
    """Positive real -> nearest E4M3 value (RTN, saturate at 448)."""
    assert x > 0
    return _round_nearest(min(x, E4M3_POS[-1]), E4M3_POS)


def e2m1_quantize(x):
    """Real -> nearest E2M1 value (RTN, ties-to-even, saturate at +/-6)."""
    mag = _round_nearest(min(abs(x), 6.0), E2M1_VALS)
    return -mag if x < 0 else mag


def e2m1_encode(value):
    """Exact E2M1 value -> 4-bit {sign, code3} nibble."""
    code3 = E2M1_MAG.index(int(abs(value) * 2))
    return (0x8 | code3) if value < 0 else code3


def quantize_block(block, M):
    """One NVFP4 block with decode scale D = e4m3(amax/M). Returns
    (D, quantized E2M1 values, squared reconstruction error)."""
    amax = max(abs(v) for v in block)
    if amax == 0:
        return 1.0, [0.0] * len(block), 0.0
    D = e4m3_quantize(amax / M)
    q = [e2m1_quantize(v / D) for v in block]
    err = sum((v - D * qi) ** 2 for v, qi in zip(block, q))
    return D, q, err


def four_over_six(block):
    """Adaptive block scaling: keep the lower-MSE of M=6 and M=4
    (strict '<' keeps the standard scale-6 on ties). Returns (D, q, M)."""
    d6, q6, e6 = quantize_block(block, 6)
    d4, q4, e4 = quantize_block(block, 4)
    return (d4, q4, 4) if e4 < e6 else (d6, q6, 6)


@cocotb.test()
async def test_nvfp4_four_over_six_dense(dut):
    """4/6 adaptive block scaling (arXiv:2512.02010) on existing
    silicon: quantize FP32 weight blocks host-side with the better of
    scale-6/scale-4, run the E2M1 codes through the dense path (4 RUNs
    of K=4 per 16-block), and assert the chip's integer partial sums
    are bit-exact AND host dequant (D/2 * sum) exactly reproduces the
    float dot against the dequantized weights."""
    start_clock(dut)
    await hw_reset(dut)

    # Paper's worked example: [10,20,30,40] scales exactly with M=4
    # (D=10, scaled {1,2,3,4} all in E2M1) but not with M=6.
    _, _, e6 = quantize_block([10.0, 20.0, 30.0, 40.0] * 4, 6)
    D, q, M = four_over_six([10.0, 20.0, 30.0, 40.0] * 4)
    assert (M, D) == (4, 10.0) and e6 > 0
    assert all(D * qi == v for qi, v in zip(q, [10.0, 20.0, 30.0, 40.0] * 4))

    # Near-max pathology: a value landing at scaled ~5 (the 4..6 gap)
    # is what 4/6 fixes — scale-4 must win on this block too.
    _, _, M5 = four_over_six([6.0, 5.0, 1.0, 2.0] + [0.0] * 12)
    assert M5 == 4

    rng = random.Random(0x4064)
    trials = 1 if GL_TEST else 3
    for t in range(trials):
        # One FP32 weight block per column; INT8 activation rows.
        blocks = [[rng.uniform(-8, 8) for _ in range(BLOCK)] for _ in range(N)]
        acts = [[rng.randint(-128, 127) for _ in range(BLOCK)] for _ in range(N)]
        quant = [four_over_six(b) for b in blocks]

        # 16-dot = 4 dense RUNs of K=4, accumulated host-side (the
        # same cross-tile accumulation any real deployment does).
        C = [[0] * N for _ in range(N)]
        for seg in range(4):
            A = [[0] * K for _ in range(N)]
            for i in range(N):
                for j in range(4):          # even slots feed dense mode
                    A[i][2 * j] = acts[i][4 * seg + j]
            wcodes = [[e2m1_encode(quant[c][1][4 * seg + j]) for j in range(4)]
                      for c in range(N)]
            part = await run_matmul(dut, A, wcodes, dense=1)
            check(dut, part, golden_dense_e2m1(A, wcodes), f"t{t} seg{seg}")
            for i in range(N):
                for c in range(N):
                    C[i][c] += part[i][c]

        # Chip sums are in the x2 domain: dequant is (D/2) * C. All
        # quantities are exact binary floats -> assert exact equality.
        for i in range(N):
            for c in range(N):
                D, q, _ = quant[c]
                ref = sum(acts[i][k] * (D * q[k]) for k in range(BLOCK))
                assert (D / 2) * C[i][c] == ref, (
                    f"t{t}: dequant C[{i}][{c}] = {(D / 2) * C[i][c]}, "
                    f"float reference = {ref}")
    dut._log.info("NVFP4 4/6 dense PASSED")


@cocotb.test()
async def test_nvfp4_four_over_six_sparse(dut):
    """Same 4/6 exactness through the 1:2-sparse path: FP32 blocks
    pruned 1:2 along k, quantized with adaptive scaling, packed as
    {select, e2m1} codes, one 16-block = 2 sparse RUNs of K=8."""
    start_clock(dut)
    await hw_reset(dut)

    rng = random.Random(0x0406)
    trials = 1 if GL_TEST else 3
    for t in range(trials):
        # 1:2-sparse FP32 blocks: one value per pair, random position.
        blocks, sels = [], []
        for _ in range(N):
            b = [0.0] * BLOCK
            s = [rng.randint(0, 1) for _ in range(BLOCK // 2)]
            for j, sel in enumerate(s):
                b[2 * j + sel] = rng.uniform(-8, 8)
            blocks.append(b)
            sels.append(s)
        acts = [[rng.randint(-128, 127) for _ in range(BLOCK)] for _ in range(N)]
        quant = [four_over_six(b) for b in blocks]

        C = [[0] * N for _ in range(N)]
        for half in range(2):               # 2 RUNs of K=8 per block
            A = [row[8 * half:8 * half + 8] for row in acts]
            wcodes = []
            for c in range(N):
                D, q, _ = quant[c]
                codes = []
                for j in range(4):
                    sel = sels[c][4 * half + j]
                    val = q[8 * half + 2 * j + sel]
                    codes.append((sel << 4) | e2m1_encode(val))
                wcodes.append(codes)
            part = await run_matmul(dut, A, wcodes)
            check(dut, part, golden_matmul(A, wcodes), f"t{t} half{half}")
            for i in range(N):
                for c in range(N):
                    C[i][c] += part[i][c]

        for i in range(N):
            for c in range(N):
                D, q, _ = quant[c]
                ref = sum(acts[i][k] * (D * q[k]) for k in range(BLOCK))
                assert (D / 2) * C[i][c] == ref, (
                    f"t{t}: sparse dequant C[{i}][{c}] = {(D / 2) * C[i][c]}, "
                    f"float reference = {ref}")
    dut._log.info("NVFP4 4/6 sparse PASSED")


@cocotb.test()
async def test_multiblock_scale_accumulation(dut):
    """Different block scales apply before cross-block accumulation.

    This is the host contract for arbitrary K: integer RUN results may be
    accumulated within one scale block, but differently-scaled blocks must
    be dequantized separately before their contributions are added.
    """
    start_clock(dut)
    await hw_reset(dut)

    rng = random.Random(0xB10C)
    # Deliberately different ranges force different E4M3 block scales.
    fp_blocks = [
        [[rng.uniform(-1, 1) for _ in range(BLOCK)] for _ in range(N)],
        [[rng.uniform(-16, 16) for _ in range(BLOCK)] for _ in range(N)],
    ]
    acts = [[rng.randint(-128, 127) for _ in range(2 * BLOCK)]
            for _ in range(N)]
    quant = [[four_over_six(fp_blocks[b][c]) for c in range(N)]
             for b in range(2)]

    integer_partials = [[[0] * N for _ in range(N)] for _ in range(2)]
    for b in range(2):
        for seg in range(4):
            A = [[0] * K for _ in range(N)]
            for i in range(N):
                for j in range(4):
                    A[i][2 * j] = acts[i][b * BLOCK + 4 * seg + j]
            wcodes = [
                [e2m1_encode(quant[b][c][1][4 * seg + j]) for j in range(4)]
                for c in range(N)
            ]
            part = await run_matmul(dut, A, wcodes, dense=1)
            for i in range(N):
                for c in range(N):
                    integer_partials[b][i][c] += part[i][c]

    saw_naive_failure = False
    for i in range(N):
        for c in range(N):
            scaled = sum((quant[b][c][0] / 2) * integer_partials[b][i][c]
                         for b in range(2))
            reference = sum(
                acts[i][b * BLOCK + k] *
                (quant[b][c][0] * quant[b][c][1][k])
                for b in range(2) for k in range(BLOCK)
            )
            assert scaled == reference, (
                f"scale-before-sum mismatch C[{i}][{c}]: "
                f"{scaled} != {reference}")

            # Demonstrate that summing integer partials first and applying
            # one block's scale is not a valid replacement.
            naive = (quant[0][c][0] / 2) * sum(
                integer_partials[b][i][c] for b in range(2))
            saw_naive_failure |= naive != reference

    assert saw_naive_failure, "test vectors did not distinguish block scales"
    dut._log.info("multi-block scale accumulation PASSED")

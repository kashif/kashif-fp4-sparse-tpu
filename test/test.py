# SPDX-FileCopyrightText: (c) 2026 Kashif
# SPDX-License-Identifier: Apache-2.0
#
# FP4 sparse mini-TPU tests.
#
# The golden model is an INDEPENDENT dense matrix multiply built from
# first principles (decode {select, e2m1} codes into a dense 6x3 matrix
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
K = 6      # contraction depth (K/2 = 3 sparse pair slots)

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
    """9 sparse codes (wcodes[col][slot]) -> dense 6x3 signed matrix."""
    W = [[0] * N for _ in range(K)]
    for c in range(N):
        for j in range(K // 2):
            sel, val = code_decode(wcodes[c][j])
            W[2 * j + sel][c] = val
    return W


def golden_matmul(A, wcodes):
    """C = A (3x6 signed INT8) x dense(W) (6x3). Exact — max |C| = 4608."""
    W = decode_weights(wcodes)
    return [[sum(A[i][k] * W[k][c] for k in range(K)) for c in range(N)]
            for i in range(N)]


def golden_dense_e2m1(A, wcodes):
    """Dense E2M1 mode: each code's nibble is a weight for ONE step;
    the select bit is ignored and only even activation slots
    (elements 0, 2, 4) participate (K=3). Exact — max |C| = 4608."""
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
    # Leave time for the clk-domain data_ready pulse and execution
    # (a RUN needs 7 cycles; this gap covers it).
    await ClockCycles(dut.clk, 12)


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
    cocotb.start_soon(Clock(dut.clk, 100, unit="ns").start())


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

@cocotb.test()
async def test_known_matmul(dut):
    """Hand-checked matmul: distinct INT8 activations, mixed E2M1 weights."""
    start_clock(dut)
    await hw_reset(dut)

    A = [[100, -128, 3, 45, 0, -2],
         [0, 17, 22, -33, 1, 127],
         [-1, -100, 2, 66, -4, 3]]
    # wcodes[col][slot] = {select, e2m1}: sel=1 puts the value at k=2j+1
    wcodes = [[0x03, 0x15, 0x02],   # col 0: W[0]=1.5(3), W[3]=3(6), W[4]=1(2)
              [0x1E, 0x04, 0x11],   # col 1: W[1]=-4(-8), W[2]=2(4), W[5]=0.5(1)
              [0x0F, 0x00, 0x17]]   # col 2: W[0]=-6(-12), (zero), W[5]=6(12)

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
    A = [[10, 20, 30, 40, 50, 60],
         [70, -80, -70, -60, -50, -40],
         [-30, -20, -10, 15, 25, 35]]

    # Single weight: E2M1 code 5 (= value 6) in col 0 slot 1 (covers k=2,3)
    for sel in (0, 1):
        wcodes = [[0x00, (sel << 4) | 0x05, 0x00],
                  [0x00, 0x00, 0x00],
                  [0x00, 0x00, 0x00]]
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

    A = [[127, -128, 99, -99, 55, -55]] * 3
    wcodes = [[0x08, 0x18, 0x08],   # -0 at every slot, both selects
              [0x08, 0x18, 0x08],
              [0x02, 0x08, 0x18]]   # one real weight: W[0]=1(2) in col 2

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

    A1 = [[10, 20, 30, 40, 50, 60]] * 3
    A2 = [[60, 50, 40, 30, 20, 10]] * 3     # same row sums, reversed order
    wcodes = [[0x01, 0x02, 0x03],           # W[0]=1, W[2]=2, W[4]=3
              [0x11, 0x12, 0x13],           # W[1]=1, W[3]=2, W[5]=3
              [0x0C, 0x00, 0x00]]           # W[0]=-4

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

    A = [[11, 12, 21, 22, 31, 32],
         [41, 42, 51, 52, 61, 62],
         [-11, -22, -33, -44, -55, -66]]
    wcodes = [[0x07, 0x17, 0x07],
              [0x16, 0x06, 0x16],
              [0x05, 0x15, 0x05]]

    expected = golden_matmul(A, wcodes)
    await load_operands(dut, A, wcodes)
    await spi_send(dut, instr_run())
    await spi_send(dut, instr_run())    # second RUN, same operands
    results = [[await read_result(dut, i, c) for c in range(N)]
               for i in range(N)]
    check(dut, results, expected, "rerun")
    dut._log.info("accumulator clear PASSED")


@cocotb.test()
async def test_dense_e2m1_mode(dut):
    """Dense E2M1 mode (RUN with dense=1): each code's nibble is a
    weight for ONE contraction step (K=3, half throughput) — dense
    INT8 x E2M1, the format NVFP4 pretraining/inference actually uses.
    Select bits are set to garbage to prove they are ignored, and odd
    activation slots hold garbage to prove they don't participate."""
    start_clock(dut)
    await hw_reset(dut)

    # Real activations at even elements 0,2,4; garbage at odd elements
    A = [[3, -88, -5, 77, 2, -18],
         [-7, 55, 6, -18, -1, 7],
         [4, -3, -128, 6, 127, -2]]

    # Full E2M1 range incl. -0; select bits (0x10) set at random
    wcodes = [[0x18, 0x07, 0x1F],   # col 0: -0, 6, -6
              [0x05, 0x1B, 0x00],   # col 1: 3, -1.5, 0
              [0x0C, 0x02, 0x19]]   # col 2: -2, 1, -0.5

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

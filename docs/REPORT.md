# FP4 Sparse Mini-TPU — Engineering Report

**Date:** 2026-07-15
**Target:** Tiny Tapeout TTSKY26c, 2x2 tile, SkyWater SKY130A
**Top module:** `tt_um_kashif_fp4_sparse_tpu`
**Clock:** 5 MHz (SPI SCLK <= clk/6)
**Result:** 7 cocotb tests passing (RTL + GL); K=6 baseline measured green
at 58.3% GPL / 50.0% effective on 2x2; this K=8 revision adds ~63 flops
(CI utilization pending)

---

## 1. Objective

Third chip of the fleet: combine the two numerics levers that the first two
chips explored separately — **1:2 structured sparsity along the contraction
axis** (from the Int7+1 chip) and the **E2M1 4-bit float element** shared by
NVFP4 and MXFP4 (from the ternary chip) — against **INT8 activations**, the
higher-precision-where-it-matters recipe the humans& NVFP4-RL blog post
names as its wished-for future format. All with no hardware multiplier.

## 2. Design Metrics

| Metric | Value |
|--------|-------|
| Array | 3x3 = 9 PEs, output-stationary systolic |
| Contraction | sparse K = 8 per RUN (dense mode K = 4); 2 sparse RUNs = one NVFP4 16-block |
| Weights | 5-bit codes {select, e2m1}: 1:2-sparse E2M1, 2.5 bits/dense position |
| Activations | INT8 (full -128..127) |
| Multiplier | None: E2M1 mag = (1\|3) << s, product = conditional 3x add + shift + negate |
| Accumulator | 14-bit signed, exact (max \|C\| = 6144 both modes) |
| Block scaling | Host-side: NVFP4 (16/E4M3+FP32), MXFP4 (32/E8M0), 4/6, per-token |
| Activation functions | Host-side (correct only after cross-tile accumulation) |
| I/O | SPI, 16-bit instructions, receive-only; results via STORE on uo_out |
| Wavefront | 8 cycles per RUN (skewed, reference pattern + one element step) |
| Tile | 2x2 (utilization TBD from CI) |

## 3. Architecture

Ported from the silicon-proven reference
([MILOUDIAS/IEEE_ttsky_mini_tpu_spi](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi))
via the Int7+1 chip's widened skeleton: operand memories, skewed wavefront
control (line i streams during t in [i+1, i+3]), activations flowing right,
weights flowing down, results accumulating in place. Both operand streams
change every cycle, so the array computes true dot products.

PE datapath per cycle: the select bit muxes one INT8 activation out of the
pair (dense mode: always the even one); the E2M1 code decodes to
`(1 or 3) << shift`, implemented as a conditional add of `act << 1`, a
2-bit shifter, and a conditional negate; the 14-bit accumulator adds the
product. All flops have async reset (no gl_preheat throwaway RUN needed,
unlike the ternary chip's no-reset pipes).

See `Architecture.drawio` and `Dataflow.drawio`.

## 4. Key Engineering Decisions

- **INT8 activations, FP4 weights (asymmetric precision).** Activations are
  harder to quantize than weights; weights carry the sparsity and the 4-bit
  element. Horizontal pipes are 16-bit INT8 pairs, vertical pipes 5-bit
  codes.
- **Shift-add instead of a multiplier.** The Int7+1 chip's 8x4 signed
  multiplier is replaced by one adder + shifter + negate — E2M1's magnitude
  set is exactly the two-mantissa-values-times-power-of-two structure that
  makes this possible.
- **5-bit weight memory.** {select, e2m1} stored as-is: 60 flops for a
  dense-equivalent 8x3 matrix — 2.5 bits per dense position.
- **14-bit exact accumulators.** 4 MAC steps x max |product| 1536 = 6144;
  no truncation, so host-side block scaling (incl. four-over-six and
  per-token scales) applies to bit-exact partial sums.
- **K deepened 6 -> 8 (one extra pair slot).** The initial design
  inherited K=6 from the reference's 3-element wavefront schedule via the
  Int7+1 chip. Adding a fourth element step costs only +63 memory flops
  (measured baseline: 58.3% GPL / 50.0% effective at K=6 on 2x2) and
  aligns the chip to NVFP4 blocks: 16 = 2 x 8, so two sparse RUNs cover
  exactly one block and the zero-padding workaround disappears. Max |acc|
  6144 still fits the 14-bit accumulator, so the datapath is unchanged.
- **Receive-only SPI** (from the ternary chip): the MISO readback stream
  duplicated the STORE readout path; dropped for area.
- **2x2 tile from day one.** The Int7+1 chip (similar flop count, bigger
  arithmetic) measured 58.7% GPL on 2x2; this design lands in the same
  class. The fleet's placement rule: <=65% effective utilization places
  cleanly, 79-95% consistently fails detailed placement.
- **CLOCK_PERIOD = 100 ns** in config.json matches the real 5 MHz clock
  (the fleet lesson: a 20 ns constraint on a 5 MHz design wastes area on
  needless timing repair).
- **Constant pins via two shared (* keep *) FFs** — avoids conb/VGND LVS
  merges (reference REPORT.md) without per-pin registers.

## 5. Verification

Golden model is an independent dense matmul built from first principles
(decode {select, e2m1} codes into a dense 8x3 matrix via the E2M1 value
table, then plain Python matrix multiply) — it shares no structure with the
RTL.

| Test | What it verifies |
|------|-----------------|
| `test_known_matmul` | Hand-checked matmul, mixed E2M1 values, full INT8 range |
| `test_select_bit_semantics` | select routes the value to k=2j or k=2j+1 |
| `test_negative_zero` | E2M1 -0 (0x8) contributes exactly zero, either select |
| `test_not_degenerate` | Equal-sum activations must differ (guards against w*sum collapse) |
| `test_run_clears_accumulators` | Back-to-back RUNs don't double |
| `test_dense_e2m1_mode` | Dense mode ignores selects + odd slots; mode latched per RUN |
| `test_random` | 12 randomized full-coverage trials, random mode per trial |

Gate-level: `GATES=yes` runs the same suite (3 random trials) against the
synthesized netlist with VPWR/VGND wiring in `tb.v`.

## 6. File Structure

```
src/
  project.v     # TT top level, SPI pin wiring, constant-pin FFs
  tpu.v         # control + memories + array + result mux
  spi.v         # 16-bit instruction receiver (receive-only)
  control.v     # LOAD/RUN/STORE decode, skewed wavefront counter
  memory_a.v    # 3x8 INT8 activation memory, pair reads
  memory_b.v    # 3x3 sparse-code weight memory (5-bit)
  array.v       # 3x3 systolic array
  pe.v          # shift-add dual-mode MAC (no multiplier)
test/
  tb.v, test.py, Makefile
docs/
  info.md, Architecture.drawio, Dataflow.drawio, REPORT.md
```

## 7. What the Chip Can Run

The chip is a W4A8 matmul primitive: 4-bit-float weights x 8-bit-integer
activations, the asymmetric precision point modern LLM deployment recipes
converge on (activations are harder to quantize than weights).

- **Off-the-shelf FP4 models (dense mode).** Any NVFP4- or MXFP4-quantized
  checkpoint maps directly: E2M1 nibbles in, INT8 activations in, exact
  integer partial sums out. No sparsification needed.
- **1:2-sparsified FP4 models (sparse mode, 2x).** Weights pruned to 1:2
  along the contraction axis (Roune's recipe: pair columns, prune+finetune)
  run at two contraction steps per cycle and 2.5 bits per dense weight
  position.
- **Arbitrary layer sizes by tiling.** Each RUN computes a 3x3 output tile
  over K=8 (dense: K=4). The host accumulates partial sums in int32 —
  bit-exact, since the chip never rounds — then applies block scales, bias,
  activation, and requantization. Block alignment is exact: two sparse
  RUNs = one NVFP4 16-element block, four = one MXFP4 32-block, with no
  padding. Four-over-six adaptive scaling and per-token activation scales
  are host-side quantizer choices and need no hardware support.
- **A bit-exactness oracle for FP4 kernels.** Because outputs are exact
  E2M1 x INT8 arithmetic, the demo board can verify software GEMM kernels
  and quantizer implementations bit-for-bit against silicon.

End-to-end demo target (shared with the fleet's pending software pipeline):
a small MNIST MLP quantized W4A8 with 1:2 weight sparsity, driven over SPI
by the RP2040 — throughput is SPI-bound, on the order of the int7 chip's
~0.15 s/digit estimate.

## 8. Companion Designs

- [kashif-int7-sparse-tpu](https://github.com/kashif/kashif-int7-sparse-tpu):
  1:2-sparse Int7 x INT4, native int8 dense mode, 3x3 on 2x2.
- [kashif-fp4-ternary-tpu](https://github.com/kashif/kashif-fp4-ternary-tpu):
  dense E2M1 x ternary (+ Bonsai INT4 mode), 4x4 on 1x2.

Together the three chips cover sparse-integer, low-bit-float x ternary, and
now sparse-float x INT8 — the full corner map of Roune's numerics argument
plus the NVFP4-RL blog's asymmetric-precision future work.

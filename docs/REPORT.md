# FP4 Sparse Mini-TPU — Engineering Report

**Date:** 2026-08-26 (serialized single-PE revision; original 3x3 array 2026-07-15)
**Target:** Tiny Tapeout TTSKY26c, 1x2 tile, SkyWater SKY130A
**Top module:** `tt_um_kashif_fp4_sparse_tpu`
**Clock:** 5 MHz (SPI SCLK <= clk/6)
**Result:** 11 cocotb tests passing; measured 70.1% GPL / 60.6%
effective on 1x2 (down from 62.5% GPL / 53.6% effective on 2x2 for the
original 9-PE parallel array — ~47% total chip area reduction)

---

## 1. Objective

Third chip of the fleet: combine the two numerics levers that the first two
chips explored separately — **1:2 structured sparsity along the contraction
axis** (from the Int7+1 chip) and the **E2M1 4-bit float element** shared by
NVFP4 and MXFP4 (from the ternary chip) — against **INT8 activations**, the
higher-precision-where-it-matters recipe the humans& NVFP4-RL blog post
names as its wished-for future format. All with no hardware multiplier.

A later revision (this document) re-architects *how* that datapath is
executed — from a spatial 9-PE systolic array down to one time-multiplexed
PE — to shrink the design from a 2x2 to a 1x2 tile. The numerics are
untouched: same E2M1 element set, same 1:2 sparsity, same K=8/dense K=4,
same exact 14-bit accumulation and the same golden arithmetic model
bit-for-bit.

## 2. Design Metrics

| Metric | Value |
|--------|-------|
| PE count | 1, time-multiplexed across all 9 (row,col) outputs |
| Contraction | sparse K = 8 per output (dense mode K = 4); 2 sparse RUNs = one NVFP4 16-block |
| Weights | 5-bit codes {select, e2m1}: 1:2-sparse E2M1, 2.5 bits/dense position |
| Activations | INT8 (full -128..127) |
| Multiplier | None: E2M1 mag = (1\|3) << s, product = conditional 3x add + shift + negate |
| Accumulator | 14-bit signed, exact (max \|C\| = 6144 both modes), per-output in result_mem |
| Block scaling | Host-side: NVFP4 (16/E4M3+FP32), MXFP4 (32/E8M0), 4/6, per-token |
| Activation functions | Host-side (correct only after cross-tile accumulation) |
| I/O | SPI, 16-bit instructions, receive-only; results via STORE on uo_out |
| RUN latency | ~45 cycles (9 outputs x (4 accumulate + 1 commit)) |
| Tile | 1x2 (measured 70.1% GPL / 60.6% effective) |

## 3. Architecture

### Why one PE instead of nine

The original design ported the silicon-proven reference
([MILOUDIAS/IEEE_ttsky_mini_tpu_spi](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi))
architecture directly: a 3x3 output-stationary systolic array, activations
flowing right, weights flowing down, a skewed 8-cycle wavefront. Measuring
real sky130 synthesis (after a separate mux-reduction pass on the memory
read ports, see §4) showed the PE array was the dominant cost regardless:
`pe.v`'s three registers (`a_reg[15:0]`, `b_reg[4:0]`, `c_reg[13:0]` = 35
bits) times 9 PEs is 315 bits — 58% of the design's entire flip-flop count
— and the shift-add datapath (mux, adder, shifter, negate) was replicated
9x on top of that.

None of that parallelism was earning its keep. A RUN's 36 operand
instructions take at least 3456 `clk` cycles to transfer at SCLK <= clk/6;
the RUN itself, even fully serialized to one PE, takes ~45 cycles — about
1.3% of the minimum load time. Spatial parallelism was free-to-remove area, not a
performance requirement.

### The new architecture

One `pe` instance, reused for every output. `control.v`'s sequencer walks
`(cur_row, cur_col)` row-major through the 9 outputs; for each, `step`
counts 0..3 across the weight-code slots, driving 4 accumulate cycles into
the PE's own `c_reg`, then a 1-cycle commit that latches the finished
value into `result_mem[cur_row][cur_col]` and clears the PE for the next
output. `pe.v` itself is byte-for-byte unchanged — same select-bit mux,
same shift-add magnitude decode, same 14-bit accumulator, same accumulator
bound (max |acc| = 4 x 1536 = 6144). `a_out`/`b_out` (the old systolic
pass-through ports, needed only by neighboring PEs) are left unconnected;
synthesis drops the now-dead forwarding registers automatically.

With no skewed wavefront to feed, `memory_a`/`memory_b` collapsed from
three concurrent per-row/column read ports (each with its own address
decode) to a single port each, addressed directly by `(row-or-col, step)`.
That was the second-largest saving — the multi-port read muxing was most
of the cost a separate memory-read-mux investigation had been chasing
(see §4); removing the need for concurrency removed the muxing itself.

Because STORE can read any output at any time after a RUN (not
necessarily right after it finishes), the design still needs somewhere to
hold all 9 finished results — `result_mem.v`, a 9x14-bit array addressed
by `(row,col)`, async-reset like `pe.v`'s old accumulators (so a STORE
issued before any RUN reads a defined 0, not X — preserving the "no
gl_preheat needed" property).

See `Architecture.drawio` and `Dataflow.drawio`.

## 4. Key Engineering Decisions

- **INT8 activations, FP4 weights (asymmetric precision).** Unchanged from
  the original design. Activations are harder to quantize than weights;
  weights carry the sparsity and the 4-bit element.
- **Shift-add instead of a multiplier.** Unchanged. E2M1's magnitude set
  is exactly the two-mantissa-values-times-power-of-two structure that
  replaces an 8x4 multiplier with one adder + shifter + negate.
- **14-bit exact accumulators.** Unchanged. 4 MAC steps x max |product|
  1536 = 6144; no truncation, so host-side block scaling (incl.
  four-over-six and per-token scales) applies to bit-exact partial sums.
- **K deepened 6 -> 8, then the array serialized 9 PEs -> 1 (this
  revision).** Two separate, independently-measured optimization passes
  on the same fleet chip:
  1. A memory-read-mux investigation found the parallel array's per-row
     addressed reads cost 17.0% of chip area in `mux2_1` cells. A
     rotate-register redesign measured *worse* (+area); an unskewed-read
     + delay-chain redesign (mirroring Google's real TPU, which has a
     genuine SRAM unified buffer) also measured worse on this
     flip-flop-only flow (+2.36% total area) — the technique needs real
     memory macros to pay off, which TinyTapeout doesn't have at this
     scale. A one-hot select redesign (control.v emits the read index
     directly as one-hot instead of round-tripping through a 2-bit
     binary encode/decode) measured a real but modest win: -1.21% total
     area, -17.2% mux area, merged to main.
  2. That still left the design at 53.6% effective utilization on 2x2 —
     roughly 2x too large for 1x2. The out-of-the-box move was to stop
     asking "how do we shrink the memory reads" and ask why the design
     needed 9 concurrent PEs at all, given RUN latency is invisible
     against SPI load time either way. Serializing to one PE (this
     document) cut total chip area ~47% (38889 -> 20760 sq-um-equivalent)
     and brought effective utilization to 28.6% on 2x2 / 60.6% on 1x2 —
     comfortably under the fleet's <=65% placement rule, confirmed by a
     real CI run (gds + gl_test + precheck, all green) at `tiles: "1x2"`.
- **Receive-only SPI** (from the ternary chip): the MISO readback stream
  duplicated the STORE readout path; dropped for area.
- **Safe SPI CDC.** The SCLK domain latches each completed word and toggles
  a one-bit completion flag. A two-flop synchronizer carries that flag into
  `clk`; the word has settled before control consumes it. CS deassertion
  discards partial frames.
- **CLOCK_PERIOD = 200 ns** in config.json matches the real 5 MHz clock.
- **Operand initialization.** Control, SPI, PE, and result state reset.
  Operand memories intentionally do not reset to save area, so software
  must load every operand it uses before the first RUN.
- **Constant pins via two shared (* keep *) FFs** — avoids conb/VGND LVS
  merges (reference REPORT.md) without per-pin registers.

## 5. Verification

Golden model is an independent dense matmul built from first principles
(decode {select, e2m1} codes into a dense 8x3 matrix via the E2M1 value
table, then plain Python matrix multiply) — it shares no structure with
the RTL, and is unchanged by the serialization (it never modeled PE
count or wavefront timing, only the {select, e2m1}-to-dense decode and
the matmul itself).

| Test | What it verifies |
|------|-----------------|
| `test_known_matmul` | Hand-checked matmul, mixed E2M1 values, full INT8 range |
| `test_select_bit_semantics` | select routes the value to k=2j or k=2j+1 |
| `test_negative_zero` | E2M1 -0 (0x8) contributes exactly zero, either select |
| `test_not_degenerate` | Equal-sum activations must differ (guards against w*sum collapse) |
| `test_run_clears_accumulators` | Back-to-back RUNs don't double — exercises `result_mem` reuse across RUNs |
| `test_spi_partial_frame_abort` | CS deassertion discards a 15-bit frame even if SCLK toggles while idle |
| `test_dense_e2m1_mode` | Dense mode ignores selects + odd slots; mode latched per RUN |
| `test_random` | 12 randomized full-coverage trials, random mode per trial |
| `test_nvfp4_four_over_six_dense` | 4/6 adaptive block scaling (arXiv:2512.02010): spec-derived NVFP4 quantizer (E4M3 RTN scales, E2M1 RTN values, MSE pick), paper's worked example, exact dequant over 4 dense RUNs |
| `test_nvfp4_four_over_six_sparse` | Same exactness through the 1:2-sparse path (one 16-block = 2 sparse RUNs) |
| `test_multiblock_scale_accumulation` | Different block scales are applied to their own integer partial sums before the block contributions are combined |

Gate-level: `GATES=yes` runs the same suite (3 random trials) against the
synthesized netlist with VPWR/VGND wiring in `tb.v`. The SPI driver's
post-instruction wait was widened from 12 to 60 cycles to cover RUN's new
~45-cycle worst case (was 8 cycles for the parallel array) — a testbench
timing parameter only; it has no effect on real SPI/clock timing.

## 6. File Structure

```
src/
  project.v     # TT top level, SPI pin wiring, constant-pin FFs
  tpu.v         # control + memories + ONE PE + result memory + result mux
  spi.v         # 16-bit instruction receiver (receive-only)
  control.v     # LOAD/RUN/STORE decode, (row,col,step) sequencer
  memory_a.v    # 3x8 INT8 activation memory, single-port read
  memory_b.v    # 3x4 sparse-code weight memory (5-bit), single-port read
  result_mem.v  # 9x14-bit result store, addressed by (row,col)
  pe.v          # shift-add dual-mode MAC (no multiplier), one instance
test/
  tb.v, test.py, Makefile
docs/
  info.md, Architecture.drawio, Dataflow.drawio, REPORT.md
```

## 7. What the Chip Can Run

The chip is a W4A8 matmul primitive: 4-bit-float weights x 8-bit-integer
activations, the asymmetric precision point modern LLM deployment recipes
converge on (activations are harder to quantize than weights). None of
this changed with the serialization — it's a pure execution-strategy and
area optimization.

- **E2M1 checkpoint weights (dense mode).** NVFP4/MXFP4 E2M1 weight
  payloads can be streamed without sparsification after the host quantizes
  activations to INT8 and manages block/tensor scales. This is W4A8
  execution, not native FP4-activation execution.
- **1:2-sparsified FP4 models (sparse mode, 2x).** Weights pruned to 1:2
  along the contraction axis (Roune's recipe: pair columns, prune+finetune)
  run at two contraction steps per weight code and 2.5 bits per dense
  weight position.
- **Arbitrary layer sizes by tiling.** Each RUN computes a 3x3 output tile
  over K=8 (dense: K=4). The host accumulates RUNs belonging to the same
  scale block in int32, applies that block's scale, and only then combines
  contributions from differently-scaled blocks. Bias, activation, and
  requantization follow the scaled sum. Block alignment is exact: two sparse
  RUNs = one NVFP4 16-element block, four = one MXFP4 32-block, with no
  padding. Four-over-six adaptive scaling and per-token activation scales
  are host-side quantizer choices and need no hardware support.
- **A bit-exactness oracle for FP4 kernels.** Because outputs are exact
  E2M1 x INT8 arithmetic, the demo board can verify software GEMM kernels
  and quantizer implementations bit-for-bit against silicon.

End-to-end demo target (shared with the fleet's pending software pipeline):
a small MNIST MLP quantized W4A8 with 1:2 weight sparsity, driven over SPI
by the RP2040 — throughput is SPI-bound regardless of PE count, on the
order of the int7 chip's ~0.15 s/digit estimate.

## 8. Companion Designs

- [kashif-int7-sparse-tpu](https://github.com/kashif/kashif-int7-sparse-tpu):
  1:2-sparse Int7 x INT4, native int8 dense mode, 3x3 parallel array on 2x2.
- [kashif-fp4-ternary-tpu](https://github.com/kashif/kashif-fp4-ternary-tpu):
  dense E2M1 x ternary (+ Bonsai INT4 mode), 4x4 parallel array on 1x2.

Together the three chips cover sparse-integer, low-bit-float x ternary, and
sparse-float x INT8 — the full corner map of Roune's numerics argument plus
the NVFP4-RL blog's asymmetric-precision future work. This chip's
serialization insight (parallelism is free to remove when SPI dominates
latency) is specific to its own architecture and hasn't been evaluated for
the other two.

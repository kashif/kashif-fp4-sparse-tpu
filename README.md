![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# FP4 Sparse Mini-TPU

Computes `C = A x W` with **1:2 structurally sparse E2M1 (NVFP4/MXFP4
element) weights**, **INT8 activations**, and **no hardware multiplier
anywhere** — using a single time-multiplexed PE instead of a 9-PE array,
since a Tiny Tapeout SPI design is completely SPI-bound: compute is
practically free, so 9x spatial parallelism buys nothing but area. Targets
a Tiny Tapeout 1x2 tile.

- [Read the documentation for project](docs/info.md)

## Why it's interesting

### Both numerics levers at once

This is the third chip in a small fleet exploring Roune's "numerics as
competitive lever" argument, and the first to combine its two big levers
in one datapath:

| Chip | Sparsity | Element type |
|------|----------|--------------|
| [Int7+1 Sparse Mini-TPU](https://github.com/kashif/kashif-int7-sparse-tpu) | 1:2 structured | Int7 |
| [NVFP4 Ternary Mini-TPU](https://github.com/kashif/kashif-fp4-ternary-tpu) | dense | E2M1 x ternary |
| **this chip** | **1:2 structured** | **E2M1 x INT8** |

It is also the asymmetric-precision recipe the
[NVFP4 RL blog post](https://humansand.ai/blog/nvfp4-rl.html) wishes for in
its future-work section: 4-bit weights against higher-precision
activations, because activations are harder to quantize than weights.

### No multiplier

E2M1 magnitudes are {0, 1, 2, 3, 4, 6, 8, 12} = (1 or 3) << shift, so each
product is one conditional add, one shift, and one conditional negate:

```
prod = ± (act [+ act << 1]) << shift
```

The most expensive cell in a conventional INT8 PE simply doesn't exist here.

### 1:2 structured sparsity in the weight format

Each 5-bit weight code `{select, e2m1}` covers TWO contraction steps; the
select bit muxes which INT8 activation of a pair enters the shift-add — an
8-bit mux does the work of a second "multiplier". Every cycle advances two
contraction steps (K=8 in the time dense does K=4), and weights store at
2.5 bits per dense position. This is the 1:2 analog of NVIDIA's 2:4
sparsity, following Roune's Int7+1 proposal with the integer swapped for
an FP4 element.

### Dense FP4 mode

The RUN flag `d` runs dense E2M1 x INT8 (K=4, half throughput, select bits
ignored) — the same trade NVIDIA makes running dense models on 2:4-sparse
tensor cores. E2M1 weight payloads from NVFP4/MXFP4 checkpoints can be
streamed after the host quantizes activations to INT8 and handles the
checkpoint's block and tensor scales; this is not native FP4-activation
execution.

| Mode | Weights | K per RUN | Rate |
|------|---------|-----------|------|
| `d=0` | 1:2-sparse E2M1 | 8 | 2 steps/cycle |
| `d=1` | dense E2M1 | 4 | 1 step/cycle |

K=8 makes the chip block-aligned: **two sparse RUNs cover exactly one NVFP4
16-element block** (four cover an MXFP4 32-block), so host-side block
scaling needs no padding.

### Exact accumulation, host-side scaling

Results are exact 14-bit signed integers (max |C| = 6144) in the x2 integer
domain — no rounding anywhere on chip. The host applies each block's scale
to that block's partial sum before combining blocks:
`C = sum_b((D_b / 2) * partial_b)`. E4M3 per 16 elements + FP32 tensor scale
(**NVFP4**), E8M0 per 32 (**MXFP4**), four-over-six adaptive scaling, and
per-token activation scales all work on these bit-exact partial sums.
Activation functions are host-side too — they are only correct after
cross-tile accumulation and bias.

### One PE, time-multiplexed across all 9 outputs

Originally a 3x3 output-stationary systolic array following the
silicon-proven [Mini-TPU v2](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi)
(operand memories, skewed 8-cycle wavefront, activations right / weights
down). Measuring real synthesis showed the 9 PEs' registers and replicated
shift-add datapath were the single largest cost in the design — 58% of
all flip-flops and most of the combinational logic — for a parallelism
the SPI-bound interface never needed: transferring all 36 operand
instructions takes at least 3456 `clk` cycles at SCLK <= clk/6
dwarfs any plausible compute latency. So the array became a single PE,
sequenced by a `(cur_row, cur_col, step)` controller through all 9
outputs, row-major, 4 accumulate cycles + 1 commit cycle each:

```
RUN issued -> for (row,col) in 9 outputs, row-major:
                for step in 0..3: acc += e2m1(W[col][step]) * A[row][step]  (4 cycles)
                result_mem[row][col] <= acc; acc <= 0                       (1 commit cycle)
              ready_to_send
```

~45 cycles total — about 1.3% of the minimum SPI load time, so the
serialization is free. The PE datapath (`pe.v`) is byte-for-byte
unchanged; only the execution strategy (spatial -> temporal) and the
memory read ports (3 concurrent -> 1 each, since there's no more skewed
wavefront to feed) changed. See `Architecture.drawio` / `Dataflow.drawio`.
Verified bit-exact against the same independent golden model the parallel
array used (`C[i][c] = sum_k A[i][k] * W[k][c]`, K = 8).

### SPI instruction set (16 bits, LSB-first; SCLK <= clk/6)

| Instruction | Format (binary)        | Description |
|-------------|------------------------|-------------|
| `LOAD A`    | `10 0 rr eee aaaaaaaa` | INT8 activation byte into row `r`, element `e` (0-7) |
| `LOAD B`    | `10 1 cc 0jj 000swwww` | Weight code {select, E2M1} into column `c`, pair slot `j` (0-3) |
| `RUN`       | `01 d 0000000000000`   | Clear accumulators, run ~45 cycles (9 outputs x (4 acc + 1 commit)); `d`=0 sparse K=8, `d`=1 dense K=4 |
| `STORE`     | `11 b rr cc 000000000` | Byte `b` (0=low, 1=high) of C[r][c] on `uo_out` |

Pins: `ui[0]`=MOSI, `ui[1]`=CS, `ui[2]`=SCLK; `uo_out`=result byte;
`uio[1]`=ready. The SPI is receive-only — results are read via STORE on
`uo_out`. MOSI, CS, and SCLK are synchronized and sampled by the 10 MHz
system clock; raw SCLK is never used as an internal clock. Keep SCLK at or
below `clk/6` and keep CS high for at least four system-clock cycles between
frames. `ready` stays high after completion until the next accepted RUN.
LOAD/STORE/RUN instructions received while a RUN is busy are ignored.

## File structure

```
src/
  project.v     # Top-level TT module (tt_um_kashif_fp4_sparse_tpu)
  tpu.v         # Core: control + memories + PE + result memory + result mux
  spi.v         # Single-clock, synchronized 16-bit SPI receiver
  control.v     # LOAD/RUN/STORE decode, (row,col,step) sequencer
  memory_a.v    # Activations: 3 rows x 8 INT8, single-port read
  memory_b.v    # Weights: 3 cols x 4 sparse codes {select, e2m1}, single-port read
  result_mem.v  # 9 x 14-bit result store, addressed by (row,col)
  pe.v          # shift-add MAC: E2M1 decode, no multiplier (one instance, reused)
test/
  tb.v          # Verilog testbench (GL_TEST compatible)
  test.py       # 12 cocotb tests with independent golden model
  pe_exhaustive_tb.sv # all 16,384 PE input/mode combinations
  control_busy_tb.sv # busy-command rejection and sticky-ready checks
  test_host.py  # host ISA/result/scaling helper tests
  Makefile      # icarus/cocotb build
docs/
  info.md, Architecture.drawio, Dataflow.drawio, REPORT.md
software/
  fp4_tpu.py    # dependency-free ISA, signed-result and scaling helpers
info.yaml       # TT metadata: 1x2 tile, 10 MHz, SKY130A
```

## Verification

12 cocotb tests drive the SPI interface like an external host and compare all
9 results against an independent golden model:

| Test | Description |
|------|-------------|
| `test_known_matmul` | Hand-checked matmul, mixed E2M1 values, full INT8 range |
| `test_select_bit_semantics` | select routes the value to k=2j or k=2j+1 |
| `test_negative_zero` | E2M1 -0 contributes exactly zero (either select) |
| `test_accumulator_extremes` | Exact signed 14-bit limits: -6144, +6144 and near-limit mixed-sign cases |
| `test_not_degenerate` | Equal-sum activations must differ (guards against w*sum collapse) |
| `test_run_clears_accumulators` | Back-to-back RUNs don't double |
| `test_spi_partial_frame_abort` | Every 0..15-bit partial frame and reset mid-frame are discarded; clk/6 transfer and sticky-ready semantics are checked |
| `test_dense_e2m1_mode` | Dense mode ignores selects and odd slots; mode latched per RUN |
| `test_random` | 12 randomized full-coverage trials, random mode |
| `test_nvfp4_four_over_six_dense` | 4/6 adaptive block scaling (arXiv:2512.02010) runs on this silicon unchanged: spec-derived NVFP4 quantizer, exact sums, exact dequant |
| `test_nvfp4_four_over_six_sparse` | Same 4/6 exactness through the 1:2-sparse path (one 16-block = 2 RUNs) |
| `test_multiblock_scale_accumulation` | Different per-block scales are applied before block contributions are combined |

Gate-level: `GATES=yes` runs the same suite (3 random trials) against the
synthesized netlist. Control, SPI, PE, and result registers have async reset,
so STORE is defined after power-up. Operand memories intentionally do not
reset to save area: software must load every operand used before the first
RUN (later RUNs may deliberately reuse loaded operands).
An additional exhaustive RTL test checks all 16,384 PE combinations across
both dense/sparse modes and select values; host-helper tests check instruction
encoding, signed result decoding, and scale-before-accumulate ordering.
A direct controller test covers busy-command rejection and sticky completion.

## Target

- **Shuttle**: TTSKY26c (SkyWater SKY130A)
- **Tile**: 1x2 (2 tiles of ~167x108 um) — production GDS, precheck, and
  gate-level simulation pass; final standard-cell utilization is 80.54%
- **Clock**: 10 MHz (SPI SCLK <= 1.67 MHz)
- **Routed timing**: zero setup/hold violations across nine corners; worst
  setup +70.8412 ns, worst hold +0.0576 ns. Maximum-slew warnings at slow
  corners remain a pre-tapeout physical-quality follow-up (see REPORT.md).

## References

- [NVFP4: NVIDIA Blackwell format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [OCP Microscaling (MX) Formats spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Structured sparsity in NVIDIA Ampere](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/)
- [The 4-bitter Lesson: NVFP4 RL](https://humansand.ai/blog/nvfp4-rl.html) — the asymmetric weight/activation precision motivation
- [Four Over Six: adaptive NVFP4 block scaling](https://arxiv.org/abs/2512.02010) — host-side compatible with this chip, validated bit-exactly by the `four_over_six` tests
- [Mini-TPU v2](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi) — architecture and SPI protocol base
- Companion designs: [Int7+1 Sparse Mini-TPU](https://github.com/kashif/kashif-int7-sparse-tpu), [NVFP4 Ternary Mini-TPU](https://github.com/kashif/kashif-fp4-ternary-tpu)
- [TT HDL Guide](https://tinytapeout.com/hdl/) / [TT Tech Specs](https://tinytapeout.com/specs/)

## What is Tiny Tapeout?

Tiny Tapeout is an educational project that aims to make it easier and cheaper than ever to get your digital and analog designs manufactured on a real chip. To learn more and get started, visit https://tinytapeout.com.

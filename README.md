![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# FP4 Sparse Mini-TPU

A 3x3 output-stationary systolic array computing `C = A x W` with **1:2
structurally sparse E2M1 (NVFP4/MXFP4 element) weights**, **INT8
activations**, and **no hardware multiplier anywhere**. Targets a Tiny
Tapeout 2x2 tile.

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
contraction steps (K=6 in the time dense does K=3), and weights store at
2.5 bits per dense position. This is the 1:2 analog of NVIDIA's 2:4
sparsity, following Roune's Int7+1 proposal with the integer swapped for
an FP4 element.

### Dense FP4 mode

The RUN flag `d` runs dense E2M1 x INT8 (K=3, half throughput, select bits
ignored) — the same trade NVIDIA makes running dense models on 2:4-sparse
tensor cores. Off-the-shelf NVFP4/MXFP4-quantized models map directly.

| Mode | Weights | K per RUN | Rate |
|------|---------|-----------|------|
| `d=0` | 1:2-sparse E2M1 | 6 | 2 steps/cycle |
| `d=1` | dense E2M1 | 3 | 1 step/cycle |

### Exact accumulation, host-side scaling

Results are exact 14-bit signed integers (max |C| = 4608) in the x2 integer
domain — no rounding anywhere on chip. The host applies whichever block
scaling it wants during dequantization: E4M3 per 16 elements + FP32 tensor
scale (**NVFP4**), E8M0 per 32 (**MXFP4**), four-over-six adaptive scaling,
and per-token activation scales all work on bit-exact partial sums.
Activation functions are host-side too — they are only correct after
cross-tile accumulation and bias.

### A real systolic matmul

The architecture is the silicon-proven
[Mini-TPU v2](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi):
operand memories, skewed 7-cycle wavefront, activations flowing right,
weights flowing down, results accumulating in place. Both operand streams
change every cycle — the array computes true dot products
(`C[i][c] = sum_k A[i][k] * W[k][c]`, K = 6), verified against an
independent golden model.

```
            W col 0    W col 1    W col 2     ({select, e2m1} codes, skewed)
               |          |          |
A row 0 --> [PE 00] -> [PE 01] -> [PE 02]     A rows: INT8 pairs
               |          |          |
A row 1 --> [PE 10] -> [PE 11] -> [PE 12]
               |          |          |
A row 2 --> [PE 20] -> [PE 21] -> [PE 22]
```

### SPI instruction set (16 bits, LSB-first; SCLK <= clk/6)

| Instruction | Format (binary)        | Description |
|-------------|------------------------|-------------|
| `LOAD A`    | `10 0 rr eee aaaaaaaa` | INT8 activation byte into row `r`, element `e` (0-5) |
| `LOAD B`    | `10 1 cc 0jj 000swwww` | Weight code {select, E2M1} into column `c`, pair slot `j` |
| `RUN`       | `01 d 0000000000000`   | Clear accumulators, run 7 cycles; `d`=0 sparse K=6, `d`=1 dense K=3 |
| `STORE`     | `11 b rr cc 000000000` | Byte `b` (0=low, 1=high) of C[r][c] on `uo_out` |

Pins: `ui[0]`=MOSI, `ui[1]`=CS, `ui[2]`=SCLK; `uo_out`=result byte;
`uio[1]`=ready. The SPI is receive-only — results are read via STORE on
`uo_out`.

## File structure

```
src/
  project.v     # Top-level TT module (tt_um_kashif_fp4_sparse_tpu)
  tpu.v         # Core: control + memories + array + result mux
  spi.v         # SPI instruction receiver, 16-bit, receive-only
  control.v     # LOAD/RUN/STORE decode, skewed wavefront counter
  memory_a.v    # Activations: 3 rows x 6 INT8, read as pairs
  memory_b.v    # Weights: 3 cols x 3 sparse codes {select, e2m1}
  array.v       # 3x3 systolic array, 14-bit exact accumulators
  pe.v          # shift-add MAC: E2M1 decode, no multiplier
test/
  tb.v          # Verilog testbench (GL_TEST compatible)
  test.py       # 7 cocotb tests with independent golden model
  Makefile      # icarus/cocotb build
docs/
  info.md, Architecture.drawio, Dataflow.drawio, REPORT.md
info.yaml       # TT metadata: 2x2 tile, 5 MHz, SKY130A
```

## Verification

7 cocotb tests drive the SPI interface like an external host and compare all
9 results against an independent golden model:

| Test | Description |
|------|-------------|
| `test_known_matmul` | Hand-checked matmul, mixed E2M1 values, full INT8 range |
| `test_select_bit_semantics` | select routes the value to k=2j or k=2j+1 |
| `test_negative_zero` | E2M1 -0 contributes exactly zero (either select) |
| `test_not_degenerate` | Equal-sum activations must differ (guards against w*sum collapse) |
| `test_run_clears_accumulators` | Back-to-back RUNs don't double |
| `test_dense_e2m1_mode` | Dense mode ignores selects and odd slots; mode latched per RUN |
| `test_random` | 12 randomized full-coverage trials, random mode |

Gate-level: `GATES=yes` runs the same suite (3 random trials) against the
synthesized netlist. All flops have async reset, so no warm-up RUN is
needed after power-up.

## Target

- **Shuttle**: TTSKY26c (SkyWater SKY130A)
- **Tile**: 2x2 (4 tiles of ~167x108 um)
- **Clock**: 5 MHz (SPI SCLK <= 833 kHz)

## References

- [NVFP4: NVIDIA Blackwell format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [OCP Microscaling (MX) Formats spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Structured sparsity in NVIDIA Ampere](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/)
- [The 4-bitter Lesson: NVFP4 RL](https://humansand.ai/blog/nvfp4-rl.html) — the asymmetric weight/activation precision motivation
- [Four Over Six: adaptive NVFP4 block scaling](https://arxiv.org/abs/2512.02010) — host-side compatible with this chip
- [Mini-TPU v2](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi) — architecture and SPI protocol base
- Companion designs: [Int7+1 Sparse Mini-TPU](https://github.com/kashif/kashif-int7-sparse-tpu), [NVFP4 Ternary Mini-TPU](https://github.com/kashif/kashif-fp4-ternary-tpu)
- [TT HDL Guide](https://tinytapeout.com/hdl/) / [TT Tech Specs](https://tinytapeout.com/specs/)

## What is Tiny Tapeout?

Tiny Tapeout is an educational project that aims to make it easier and cheaper than ever to get your digital and analog designs manufactured on a real chip. To learn more and get started, visit https://tinytapeout.com.

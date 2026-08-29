/*
 * TT FP4 Sparse Mini-TPU — Tiny Tapeout top level (1x2 tile)
 *
 * One time-multiplexed PE computing a 3x3 C = A x W output tile with
 * INT8 activations and 1:2-sparse E2M1 (NVFP4/MXFP4 element) weights:
 * each 5-bit weight code {select, e2m1[3:0]} covers two consecutive
 * contraction steps, the select bit muxing which INT8 activation of a
 * pair gets multiplied. There is no hardware multiplier — E2M1
 * magnitudes are (1 or 3) << shift, so each product is a conditional
 * 3x add, a shift, and a conditional negate.
 *
 * Architecture and SPI protocol follow the proven reference design
 * (github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi), widened to 16-bit
 * instructions. See docs/info.md for the ISA.
 *
 * Pinout:
 *   ui_in[0] = MOSI, ui_in[1] = CS (active low), ui_in[2] = SCLK
 *   uo_out   = result byte (selected by STORE)
 *   uio[1]   = ready (sticky until next RUN), rest unused (SPI is receive-only;
 *              results are read via STORE on uo_out)
 */

`default_nettype none

module tt_um_kashif_fp4_sparse_tpu (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    wire        mosi = ui_in[0];
    wire        cs   = ui_in[1];
    wire        sclk = ui_in[2];

    wire [15:0] instruction;
    wire        ready_to_send;

    tpu u_tpu (
        .clk            (clk),
        .rst_n          (rst_n),
        .instruction    (instruction),
        .ready_to_send  (ready_to_send),
        .result         (uo_out)
    );

    spi u_spi (
        .clk                (clk),
        .rst_n              (rst_n),
        .mosi               (mosi),
        .cs                 (cs),
        .sclk               (sclk),
        .data_buffer_output (instruction)
    );

    // Constant pins are driven from two shared (* keep *) flip-flops
    // instead of per-pin registers or direct constants: constants
    // synthesize to conb cells whose pulldown nets magic merges with
    // VGND during extraction, producing LVS mismatches (reference
    // REPORT.md), and per-pin FFs waste flops. uio[1] = ready out,
    // everything else stays an input (oe 0).
    (* keep = "true" *) reg const0_q, const1_q;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            const0_q <= 1'b0;
            const1_q <= 1'b0;
        end else begin
            const0_q <= 1'b0;
            const1_q <= 1'b1;
        end
    end
    assign uio_oe  = {{6{const0_q}}, const1_q, const0_q};
    assign uio_out = {{6{const0_q}}, ready_to_send, const0_q};

    wire _unused = &{ena, ui_in[7:3], uio_in, 1'b0};

endmodule

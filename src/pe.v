/*
 * Processing Element: 1:2-sparse E2M1 x INT8 MAC (output-stationary),
 * multiplier-free.
 *
 * Dataflow follows the reference mini-TPU PE (activations flow right,
 * weights flow down). The weight pipe carries a 5-bit code
 * {select, sign, mag[2:0]} — an E2M1 (NVFP4/MXFP4 element) nibble
 * plus a sparsity select bit. Two modes, chosen by `dense`:
 *
 *  Sparse (dense=0): the code covers TWO consecutive contraction
 *    steps (1:2 structured sparsity along k). The activation pipe
 *    carries an INT8 pair and the select bit muxes which one gets
 *    multiplied — an 8-bit mux does the work of a second "multiplier",
 *    so each cycle advances two steps of the contraction.
 *
 *  Dense (dense=1): the code is a weight for ONE contraction step;
 *    only the even activation of the pair is used. Half throughput,
 *    same hardware — dense E2M1-weight x INT8-activation operation.
 *
 * There is no multiplier: E2M1 magnitudes are {0,1,2,3,4,6,8,12} =
 * (1 or 3) << shift, so the product is a conditional 3x (one add),
 * a shift, and a conditional negate.
 *
 * The PE flops have async reset (dfrtp) — avoids the gate-level
 * X-poisoning documented in the reference REPORT.md for no-reset
 * dfxtp pipeline registers.
 */

`default_nettype none

module pe (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        we,      // shift pipes + accumulate
    input  wire        clr,     // clear accumulator (start of RUN)
    input  wire        dense,   // 0 = 1:2 sparse (K=8), 1 = dense (K=4)
    input  wire [15:0] a_in,    // activation pair {a_odd[7:0], a_even[7:0]}
    input  wire [4:0]  b_in,    // weight code {select, sign, mag[2:0]}
    output wire [15:0] a_out,   // pair passed right
    output wire [4:0]  b_out,   // code passed down
    output wire [13:0] c_out    // accumulated result
);

    reg [15:0] a_reg;
    reg [4:0]  b_reg;
    reg signed [13:0] c_reg;

    wire       select = b_in[4];
    wire       w_sign = b_in[3];
    wire [2:0] code   = b_in[2:0];

    // Activation: dense always uses the even slot; sparse muxes by select.
    wire signed [7:0] act = (dense || !select) ? $signed(a_in[7:0])
                                               : $signed(a_in[15:8]);

    // E2M1 magnitude = m << s with m in {1,3}:
    //   code:  1  2  3  4  5  6  7      (0 = zero)
    //   mag:   1  2  3  4  6  8  12
    //   m3:    0  0  1  0  1  0  1
    //   s:     0  1  0  2  1  3  2
    wire       m3    = code[0] && (code[2:1] != 2'b0);
    wire [1:0] shamt = m3 ? (code[2:1] - 2'd1) : code[2:1];

    // |act| <= 128: m3 path max 384 << 2 = 1536, plain path max
    // 128 << 3 = 1024 — both fit signed 12 bits, as does the negate.
    wire signed [11:0] act_w = {{4{act[7]}}, act};
    wire signed [11:0] base  = m3 ? (act_w + (act_w <<< 1)) : act_w;
    wire signed [11:0] mag   = base <<< shamt;
    wire signed [11:0] prod  = (code == 3'd0) ? 12'sd0
                             : (w_sign ? -mag : mag);

    // 4 MAC steps either mode (each sparse code contributes ONE
    // product; the other k of the pair is zero): max |acc| =
    // 4 * 1536 = 6144 — exact in a 14-bit signed accumulator.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_reg <= 16'd0;
            b_reg <= 5'd0;
            c_reg <= 14'sd0;
        end else begin
            if (clr)
                c_reg <= 14'sd0;
            else if (we)
                c_reg <= c_reg + prod;
            if (we) begin
                a_reg <= a_in;
                b_reg <= b_in;
            end
        end
    end

    assign a_out = a_reg;
    assign b_out = b_reg;
    assign c_out = c_reg;

endmodule

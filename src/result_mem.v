/*
 * Result memory: 9 accumulators (3x3, row-major), 14-bit signed each.
 *
 * Replaces the array's 9 per-PE accumulator registers with a single
 * addressed store, written once per (row,col) output as the
 * time-multiplexed PE finishes it (see control.v, pe_serial wiring
 * in tpu.v) and read back by STORE at any time after -- the same
 * "any output, any time" contract the parallel array offered.
 *
 * Async reset (matches pe.v's c_reg): a STORE issued before any RUN
 * reads a defined 0, not X -- avoiding the gl_preheat/no-reset
 * X-poisoning class of gate-level bug documented in the reference
 * REPORT.md.
 */

`default_nettype none

module result_mem (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              write_enable,
    input  wire [1:0]        write_row,
    input  wire [1:0]        write_col,
    input  wire signed [13:0] write_data,
    input  wire [1:0]        read_row,
    input  wire [1:0]        read_col,
    output wire signed [13:0] read_data
);

    reg signed [13:0] acc [0:2][0:2];

    genvar r, c;
    generate
        for (r = 0; r < 3; r = r + 1) begin : row
            for (c = 0; c < 3; c = c + 1) begin : col
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n)
                        acc[r][c] <= 14'sd0;
                    else if (write_enable && write_row == r[1:0] && write_col == c[1:0])
                        acc[r][c] <= write_data;
                end
            end
        end
    endgenerate

    assign read_data = (read_row < 2'd3 && read_col < 2'd3)
        ? acc[read_row][read_col]
        : 14'sd0;

endmodule

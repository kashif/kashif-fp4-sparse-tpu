/*
 * Activation memory: 3 rows x 8 INT8 elements (K = 8 contraction depth).
 *
 * Written one byte at a time (elem 0..7). Read out as an INT8 *pair*
 * {A[row][2j+1], A[row][2j]} through a SINGLE port, addressed by
 * (row, pair) -- the time-multiplexed single PE only ever computes
 * one (row,col) output at a time (see control.v), so there is no
 * skewed wavefront and no need for concurrent per-row read ports
 * (the thing that made the old parallel-array design's read muxing
 * expensive). Output is 0 when not enabled.
 */

`default_nettype none

module memory_a (
    input  wire        clk,
    input  wire        write_enable,
    input  wire [1:0]  write_line,      // row 0..2
    input  wire [2:0]  write_elem,      // element 0..7
    input  wire [7:0]  data_in,
    input  wire        read_enable,
    input  wire [1:0]  read_row,        // row 0..2
    input  wire [1:0]  read_pair,       // pair slot 0..3
    output wire [15:0] data_out         // one 16-bit pair
);

    reg [7:0] mem [0:2][0:7];

    always @(posedge clk) begin
        if (write_enable && write_line < 2'd3)
            mem[write_line][write_elem] <= data_in;
    end

    assign data_out = read_enable
        ? {mem[read_row][{read_pair, 1'b1}], mem[read_row][{read_pair, 1'b0}]}
        : 16'd0;

endmodule

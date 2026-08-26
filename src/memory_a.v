/*
 * Activation memory: 3 rows x 8 INT8 elements (K = 8 contraction depth).
 *
 * Written one byte at a time (elem 0..7). Read out per row as an
 * INT8 *pair* {A[row][2j+1], A[row][2j]} selected by a one-hot
 * 4-bit code (which of the 4 pair slots is live this cycle) -- the
 * systolic array consumes two contraction steps per cycle in sparse
 * mode. Rows read as 0 when not enabled (feeds zeros outside the
 * wavefront). control.v produces the select one-hot directly from
 * its counter comparisons, so there is no binary encode/decode round
 * trip between control and memory.
 */

`default_nettype none

module memory_a (
    input  wire        clk,
    input  wire        write_enable,
    input  wire [1:0]  write_line,      // row 0..2
    input  wire [2:0]  write_elem,      // element 0..7
    input  wire [7:0]  data_in,
    input  wire [2:0]  read_enable,     // per-row
    input  wire [11:0] read_sel,        // one-hot 4-bit pair select per row
    output wire [47:0] data_out         // 3 rows x 16-bit pair
);

    reg [7:0] mem [0:2][0:7];

    always @(posedge clk) begin
        if (write_enable && write_line < 2'd3)
            mem[write_line][write_elem] <= data_in;
    end

    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : read_row
            wire [3:0] sel = read_sel[4*i +: 4];
            wire [15:0] pair0 = {mem[i][1], mem[i][0]};
            wire [15:0] pair1 = {mem[i][3], mem[i][2]};
            wire [15:0] pair2 = {mem[i][5], mem[i][4]};
            wire [15:0] pair3 = {mem[i][7], mem[i][6]};
            assign data_out[16*i +: 16] = read_enable[i]
                ? ({16{sel[0]}} & pair0) | ({16{sel[1]}} & pair1) |
                  ({16{sel[2]}} & pair2) | ({16{sel[3]}} & pair3)
                : 16'd0;
        end
    endgenerate

endmodule

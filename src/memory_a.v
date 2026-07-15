/*
 * Activation memory: 3 rows x 6 INT8 elements (K = 6 contraction depth).
 *
 * Written one byte at a time (elem 0..5). Read out per row as an
 * INT8 *pair* {A[row][2j+1], A[row][2j]} selected by a 2-bit pair
 * index — the systolic array consumes two contraction steps per cycle
 * in sparse mode. Rows read as 0 when not enabled (feeds zeros
 * outside the wavefront).
 */

`default_nettype none

module memory_a (
    input  wire        clk,
    input  wire        write_enable,
    input  wire [1:0]  write_line,      // row 0..2
    input  wire [2:0]  write_elem,      // element 0..5
    input  wire [7:0]  data_in,
    input  wire [2:0]  read_enable,     // per-row
    input  wire [5:0]  read_pair,       // 2-bit pair index per row (0..2)
    output wire [47:0] data_out         // 3 rows x 16-bit pair
);

    reg [7:0] mem [0:2][0:5];

    always @(posedge clk) begin
        if (write_enable && write_line < 2'd3 && write_elem < 3'd6)
            mem[write_line][write_elem] <= data_in;
    end

    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : read_row
            wire [1:0] pair = read_pair[2*i +: 2];
            assign data_out[16*i +: 16] = read_enable[i]
                ? {mem[i][{pair, 1'b1}], mem[i][{pair, 1'b0}]}
                : 16'd0;
        end
    endgenerate

endmodule

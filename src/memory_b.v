/*
 * Weight memory: 3 columns x 4 sparse-E2M1 codes (5 bits each).
 *
 * Each code {select, e2m1[3:0]} covers TWO dense weight positions
 * (contraction steps k=2j and k=2j+1 of column c): the E2M1 nibble
 * sits at k = 2j + select, the other position is zero. 12 codes
 * encode a dense 8x3 weight matrix in 60 bits -- 2.5 bits per dense
 * position, the storage saving of 1:2 sparsity on a 4-bit element.
 *
 * Read out per column, one code per wavefront step, selected by a
 * one-hot 4-bit code from control.v (see memory_a.v). Columns read
 * as 0 when not enabled (code 0 decodes to +0 -> product 0).
 */

`default_nettype none

module memory_b (
    input  wire        clk,
    input  wire        write_enable,
    input  wire [1:0]  write_line,      // column 0..2
    input  wire [1:0]  write_elem,      // pair slot 0..3
    input  wire [4:0]  data_in,
    input  wire [2:0]  read_enable,     // per-column
    input  wire [11:0] read_sel,        // one-hot 4-bit slot select per column
    output wire [14:0] data_out         // 3 cols x 5-bit code
);

    reg [4:0] mem [0:2][0:3];

    always @(posedge clk) begin
        if (write_enable && write_line < 2'd3)
            mem[write_line][write_elem] <= data_in;
    end

    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : read_col
            wire [3:0] sel = read_sel[4*i +: 4];
            assign data_out[5*i +: 5] = read_enable[i]
                ? ({5{sel[0]}} & mem[i][0]) | ({5{sel[1]}} & mem[i][1]) |
                  ({5{sel[2]}} & mem[i][2]) | ({5{sel[3]}} & mem[i][3])
                : 5'd0;
        end
    endgenerate

endmodule

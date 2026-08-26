/*
 * Weight memory: 3 columns x 4 sparse-E2M1 codes (5 bits each).
 *
 * Each code {select, e2m1[3:0]} covers TWO dense weight positions
 * (contraction steps k=2j and k=2j+1 of column c): the E2M1 nibble
 * sits at k = 2j + select, the other position is zero. 12 codes
 * encode a dense 8x3 weight matrix in 60 bits -- 2.5 bits per dense
 * position, the storage saving of 1:2 sparsity on a 4-bit element.
 *
 * Read out UNSKEWED: all 3 columns are read simultaneously, one
 * shared 2-bit slot index -- the per-column launch stagger is
 * reconstructed downstream by skew3.v, not by addressing here.
 * Output is 0 outside the shared 4-cycle read window (code 0 also
 * decodes to +0 -> product 0, belt and braces).
 */

`default_nettype none

module memory_b (
    input  wire        clk,
    input  wire        write_enable,
    input  wire [1:0]  write_line,      // column 0..2
    input  wire [1:0]  write_elem,      // pair slot 0..3
    input  wire [4:0]  data_in,
    input  wire        read_enable,     // shared across all 3 columns
    input  wire [1:0]  read_slot,       // shared 2-bit slot index (0..3)
    output wire [14:0] data_out         // 3 cols x 5-bit code, unskewed
);

    reg [4:0] mem [0:2][0:3];

    always @(posedge clk) begin
        if (write_enable && write_line < 2'd3)
            mem[write_line][write_elem] <= data_in;
    end

    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : read_col
            assign data_out[5*i +: 5] = read_enable
                ? mem[i][read_slot]
                : 5'd0;
        end
    endgenerate

endmodule

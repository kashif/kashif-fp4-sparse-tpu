/*
 * Activation memory: 3 rows x 8 INT8 elements (K = 8 contraction depth).
 *
 * Written one byte at a time (elem 0..7). Read out UNSKEWED: all 3
 * rows are read simultaneously as an INT8 *pair* {A[row][2j+1],
 * A[row][2j]}, selected by one shared 2-bit pair index -- the
 * per-row systolic launch stagger is reconstructed downstream by
 * skew3.v (a plain delay chain), not by addressing here. Output is
 * 0 outside the shared 4-cycle read window.
 */

`default_nettype none

module memory_a (
    input  wire        clk,
    input  wire        write_enable,
    input  wire [1:0]  write_line,      // row 0..2
    input  wire [2:0]  write_elem,      // element 0..7
    input  wire [7:0]  data_in,
    input  wire        read_enable,     // shared across all 3 rows
    input  wire [1:0]  read_pair,       // shared 2-bit pair index (0..3)
    output wire [47:0] data_out         // 3 rows x 16-bit pair, unskewed
);

    reg [7:0] mem [0:2][0:7];

    always @(posedge clk) begin
        if (write_enable && write_line < 2'd3)
            mem[write_line][write_elem] <= data_in;
    end

    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : read_row
            assign data_out[16*i +: 16] = read_enable
                ? {mem[i][{read_pair, 1'b1}], mem[i][{read_pair, 1'b0}]}
                : 16'd0;
        end
    endgenerate

endmodule

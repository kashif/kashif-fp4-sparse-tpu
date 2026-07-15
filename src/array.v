/*
 * 3x3 output-stationary systolic array, 1:2-sparse E2M1 x INT8 edition.
 *
 * Structure follows the reference mini-TPU array.v: activations flow
 * right, weights flow down, results accumulate in place. Differences:
 *   - horizontal pipes carry an INT8 *pair* (16 bits) — two
 *     contraction steps per cycle in sparse mode
 *   - vertical pipes carry 5-bit weight codes {select, e2m1[3:0]}
 *   - 14-bit accumulators (exact, no truncation)
 *   - clr input zeroes all accumulators at the start of a RUN
 */

`default_nettype none

module array (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         we,
    input  wire         clr,
    input  wire         dense,    // 0 = 1:2 sparse (K=6), 1 = dense (K=3)

    input  wire [47:0]  a_in,     // 3 rows x INT8 activation pair
    input  wire [14:0]  b_in,     // 3 cols x 5-bit weight code
    output wire [125:0] data_out  // 9 accumulators x 14 bits, row-major
);

    // a_pipe[row][col]: pair flowing right; b_pipe[row][col]: code flowing down
    wire [15:0] a_pipe [0:2][0:3];
    wire [4:0]  b_pipe [0:3][0:2];
    wire [13:0] c_bus  [0:2][0:2];

    genvar row, col;
    generate
        for (row = 0; row < 3; row = row + 1) begin : map_a_in
            assign a_pipe[row][0] = a_in[16*row +: 16];
        end
        for (col = 0; col < 3; col = col + 1) begin : map_b_in
            assign b_pipe[0][col] = b_in[5*col +: 5];
        end
    endgenerate

    generate
        for (row = 0; row < 3; row = row + 1) begin : ROWS
            for (col = 0; col < 3; col = col + 1) begin : COLS
                pe pe_inst (
                    .clk   (clk),
                    .rst_n (rst_n),
                    .we    (we),
                    .clr   (clr),
                    .dense (dense),
                    .a_in  (a_pipe[row][col]),
                    .b_in  (b_pipe[row][col]),
                    .a_out (a_pipe[row][col+1]),
                    .b_out (b_pipe[row+1][col]),
                    .c_out (c_bus [row][col])
                );
            end
        end
    endgenerate

    generate
        for (row = 0; row < 3; row = row + 1) begin : flat_row
            for (col = 0; col < 3; col = col + 1) begin : flat_col
                assign data_out[14*(row*3+col) +: 14] = c_bus[row][col];
            end
        end
    endgenerate

endmodule

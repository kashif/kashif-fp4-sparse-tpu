/*
 * Weight memory: 3 columns x 4 sparse-E2M1 codes (5 bits each).
 *
 * Each code {select, e2m1[3:0]} covers TWO dense weight positions
 * (contraction steps k=2j and k=2j+1 of column c): the E2M1 nibble
 * sits at k = 2j + select, the other position is zero. 12 codes
 * encode a dense 8x3 weight matrix in 60 bits -- 2.5 bits per dense
 * position, the storage saving of 1:2 sparsity on a 4-bit element.
 *
 * Read out per column, one code per wavefront step, always slot
 * order 0,1,2,3 -- never random access. Same circular-rotate
 * structure as memory_a (see its header for the reuse-across-RUNs
 * argument): the read port is always slot 0, one rotation per active
 * cycle, no address-decoded mux. Columns read as 0 when not enabled
 * (code 0 decodes to +0 -> product 0).
 */

`default_nettype none

module memory_b (
    input  wire        clk,
    input  wire        write_enable,
    input  wire [1:0]  write_line,      // column 0..2
    input  wire [1:0]  write_elem,      // pair slot 0..3
    input  wire [4:0]  data_in,
    input  wire [2:0]  read_enable,     // per-column; also drives auto-rotate
    output wire [14:0] data_out         // 3 cols x 5-bit code
);

    reg [4:0] q [0:2][0:3];

    genvar i, s;
    generate
        for (i = 0; i < 3; i = i + 1) begin : col
            for (s = 0; s < 4; s = s + 1) begin : slot
                always @(posedge clk) begin
                    if (write_enable && write_line == i[1:0] && write_elem == s[1:0])
                        q[i][s] <= data_in;
                    else if (read_enable[i])
                        q[i][s] <= q[i][(s + 1) % 4];
                end
            end

            assign data_out[5*i +: 5] = read_enable[i] ? q[i][0] : 5'd0;
        end
    endgenerate

endmodule

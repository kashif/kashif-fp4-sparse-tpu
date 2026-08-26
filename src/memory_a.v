/*
 * Activation memory: 3 rows x 8 INT8 elements (K = 8 contraction depth).
 *
 * Written one byte at a time (elem 0..7). Read out per row as an
 * INT8 *pair* {A[row][2j+1], A[row][2j]} -- the systolic array
 * consumes two contraction steps per cycle in sparse mode. Rows read
 * as 0 when not enabled (feeds zeros outside the wavefront).
 *
 * The skewed wavefront always walks a row's 4 pairs in the fixed
 * order 0,1,2,3, one per active cycle -- never random access. Rather
 * than a live 4:1 address-decoded mux at the read port (what a flat
 * 8-byte array with a computed index synthesizes to), each row is
 * stored as 4 pair-slots in a circular rotate register: the read is
 * always slot 0, and one rotation per active cycle brings the next
 * pair into place. A row's window is exactly 4 active cycles, so
 * rotation always completes a full circle and lands back on natural
 * order -- the same operands are correctly reusable across multiple
 * RUNs. Writes must not overlap an in-flight RUN reading the same
 * row (true of the SPI protocol already: the host waits for `ready`
 * before the next instruction), so a write always lands while the
 * row is in natural order and targets the semantically correct slot.
 */

`default_nettype none

module memory_a (
    input  wire        clk,
    input  wire        write_enable,
    input  wire [1:0]  write_line,      // row 0..2
    input  wire [2:0]  write_elem,      // element 0..7
    input  wire [7:0]  data_in,
    input  wire [2:0]  read_enable,     // per-row; also drives auto-rotate
    output wire [47:0] data_out         // 3 rows x 16-bit pair
);

    reg [15:0] q [0:2][0:3];

    genvar i, s;
    generate
        for (i = 0; i < 3; i = i + 1) begin : row
            for (s = 0; s < 4; s = s + 1) begin : pair_slot
                always @(posedge clk) begin
                    if (write_enable && write_line == i[1:0] && write_elem[2:1] == s[1:0]) begin
                        if (write_elem[0])
                            q[i][s][15:8] <= data_in;
                        else
                            q[i][s][7:0]  <= data_in;
                    end else if (read_enable[i]) begin
                        q[i][s] <= q[i][(s + 1) % 4];
                    end
                end
            end

            assign data_out[16*i +: 16] = read_enable[i] ? q[i][0] : 16'd0;
        end
    endgenerate

endmodule

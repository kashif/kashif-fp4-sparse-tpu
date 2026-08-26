/*
 * 3-lane diagonal delay chain for systolic operand skew.
 *
 * Feeding a systolic row/column from a shared, unskewed memory read
 * needs the launch-time stagger reconstructed downstream: lane 0
 * passes straight through, lane 1 is delayed one cycle, lane 2 two
 * cycles -- plain shift registers, no addressing, no select logic at
 * all (Google's TPU applies the same "unskewed read + diagonal delay
 * chain" pattern in its systolic data setup unit, ahead of a real
 * SRAM unified buffer -- here it sits after our flip-flop memories
 * instead).
 */

`default_nettype none

module skew3 #(
    parameter LANE_WIDTH = 16
) (
    input  wire                    clk,
    input  wire [3*LANE_WIDTH-1:0] in,   // lanes 0,1,2 concatenated, unskewed
    output wire [3*LANE_WIDTH-1:0] out   // lane i delayed by i cycles
);

    reg [LANE_WIDTH-1:0] d1;        // 1-cycle delay, lane 1
    reg [LANE_WIDTH-1:0] d2a, d2b;  // 2-cycle delay, lane 2

    always @(posedge clk) begin
        d1  <= in[LANE_WIDTH   +: LANE_WIDTH];
        d2a <= in[2*LANE_WIDTH +: LANE_WIDTH];
        d2b <= d2a;
    end

    assign out[0            +: LANE_WIDTH] = in[0 +: LANE_WIDTH];  // lane 0: passthrough
    assign out[LANE_WIDTH   +: LANE_WIDTH] = d1;                    // lane 1: 1-cycle
    assign out[2*LANE_WIDTH +: LANE_WIDTH] = d2b;                   // lane 2: 2-cycle

endmodule

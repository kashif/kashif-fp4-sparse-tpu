/*
 * Control unit — ported from the reference mini-TPU control.v,
 * widened to a 16-bit instruction and extended with accumulator
 * clear and 2-byte result readout.
 *
 * Instruction format (16 bits, sent LSB-first over SPI):
 *
 *   [15:14] opcode: 00=NOP, 01=RUN, 10=LOAD, 11=STORE
 *
 *   LOAD:  [13]    mem_select (0 = A activations, 1 = B weights)
 *          [12:11] line  (A: row 0-2, B: column 0-2)
 *          [10:8]  elem  (A: element 0-7, B: pair slot 0-3)
 *          [7:0]   imm   (A: INT8 byte, B: {select, e2m1} in imm[4:0])
 *
 *   RUN:   [13] dense_mode (0 = 1:2 sparse K=8, 1 = dense E2M1 K=4)
 *          Clears accumulators, streams the skewed wavefront for
 *          8 cycles (4 weight-code steps either way — sparse covers
 *          two contraction steps per code, dense covers one).
 *
 *   STORE: [13]    byte_sel (0 = acc[7:0], 1 = {2'b0, acc[13:8]})
 *          [12:11] row, [10:9] col
 *          Latches the selection; result output holds until next STORE.
 *
 * Memories A and B share the same skewed read pattern (line i active
 * during counter in [i+1, i+4], element walking 0,1,2,3) — the
 * reference schedule with one extra element step; one sparse pair
 * step has the same schedule as one dense step, which is exactly the
 * 2x throughput claim. Two sparse RUNs cover exactly one NVFP4
 * 16-element block (K=8 dense-equivalent per RUN).
 */

`default_nettype none

module control (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] instruction,

    output wire        array_write_enable,
    output wire        array_clear,
    output reg         dense_mode,
    output reg  [1:0]  store_row,
    output reg  [1:0]  store_col,
    output reg         store_byte_sel,

    output wire [7:0]  mema_data_in,
    output wire        mema_write_enable,
    output wire [1:0]  mema_write_line,
    output wire [2:0]  mema_write_elem,
    output wire [2:0]  mema_read_enable,
    output wire [5:0]  mema_read_elem,

    output wire [4:0]  memb_data_in,
    output wire        memb_write_enable,
    output wire [1:0]  memb_write_line,
    output wire [1:0]  memb_write_elem,
    output wire [2:0]  memb_read_enable,
    output wire [5:0]  memb_read_elem,

    output reg         ready_to_send
);

    localparam [1:0] RUN   = 2'b01;
    localparam [1:0] LOAD  = 2'b10;
    localparam [1:0] STORE = 2'b11;

    // Wavefront counter: useful range 1..8 (last product forms at
    // r+c+1+j = 2+2+1+3 = 8)
    reg [3:0] counter;

    wire [1:0] opcode     = instruction[15:14];
    wire       mem_select = instruction[13];
    wire [1:0] line       = instruction[12:11];
    wire [2:0] elem       = instruction[10:8];
    wire [7:0] imm        = instruction[7:0];

    wire is_load  = (opcode == LOAD);
    wire is_store = (opcode == STORE);
    wire is_run   = (opcode == RUN) || (counter > 0);

    // Latch the weight format on each RUN; stationary during the wavefront.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            dense_mode <= 1'b0;
        else if (opcode == RUN && counter == 4'd0)
            dense_mode <= instruction[13];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter       <= 4'd0;
            ready_to_send <= 1'b0;
        end else if (counter == 4'd8) begin
            counter       <= 4'd0;
            ready_to_send <= 1'b1;
        end else if (is_run)
            counter <= counter + 4'd1;
        else
            ready_to_send <= 1'b0;
    end

    // ------------------------------------------------------------------
    // Shared skewed read pattern (identical for A rows and B columns):
    // line i streams its 4 elements during counter in [i+1, i+4].
    // ------------------------------------------------------------------
    wire [2:0] read_enable_shared;
    wire [5:0] read_elem_shared;

    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : read_pattern_gen
            assign read_enable_shared[i] = (counter > i) && (counter < (i + 5));
            assign read_elem_shared[2*i +: 2] =
                (counter == (i + 1)) ? 2'd0 :
                (counter == (i + 2)) ? 2'd1 :
                (counter == (i + 3)) ? 2'd2 :
                (counter == (i + 4)) ? 2'd3 : 2'd0;
        end
    endgenerate

    assign mema_read_enable = read_enable_shared;
    assign memb_read_enable = read_enable_shared;
    assign mema_read_elem   = read_elem_shared;
    assign memb_read_elem   = read_elem_shared;

    // ------------------------------------------------------------------
    // Write path
    // ------------------------------------------------------------------
    wire load_a = is_load && !mem_select;
    wire load_b = is_load &&  mem_select;

    assign mema_data_in      = imm;
    assign mema_write_enable = load_a;
    assign mema_write_line   = line;
    assign mema_write_elem   = elem;

    assign memb_data_in      = imm[4:0];
    assign memb_write_enable = load_b;
    assign memb_write_line   = line;
    assign memb_write_elem   = elem[1:0];

    // ------------------------------------------------------------------
    // STORE: latch result selection so uo_out holds a stable value
    // between SPI transactions (instruction is a 1-cycle pulse).
    // ------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            store_row      <= 2'd0;
            store_col      <= 2'd0;
            store_byte_sel <= 1'b0;
        end else if (is_store) begin
            store_row      <= instruction[12:11];
            store_col      <= instruction[10:9];
            store_byte_sel <= instruction[13];
        end
    end

    assign array_write_enable = is_run;
    // Clear accumulators on the RUN-issue cycle; the wavefront reads are
    // still disabled then (counter == 0), so no products are lost.
    assign array_clear        = (opcode == RUN) && (counter == 4'd0);

endmodule

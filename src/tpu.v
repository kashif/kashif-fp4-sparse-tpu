/*
 * 1:2-sparse E2M1 x INT8 mini-TPU core: control + operand memories +
 * 3x3 systolic array + result readout mux.
 *
 * Computes C = A x W with A a 3x8 INT8 activation matrix and W a
 * dense-equivalent 8x3 E2M1 weight matrix stored as 12 five-bit codes
 * {select, e2m1} (1:2 structured sparsity along the contraction
 * axis). Results are exact 14-bit signed values in the x2 integer
 * domain, read out one byte at a time via STORE.
 */

`default_nettype none

module tpu (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [15:0] instruction,
    output wire        ready_to_send,
    output wire [7:0]  result
);

    wire        array_write_enable;
    wire        array_clear;
    wire        dense_mode;
    wire [1:0]  store_row, store_col;
    wire        store_byte_sel;

    wire [7:0]  mema_data_in;
    wire        mema_write_enable;
    wire [1:0]  mema_write_line;
    wire [2:0]  mema_write_elem;
    wire [2:0]  mema_read_enable;
    wire [11:0] mema_read_sel;

    wire [4:0]  memb_data_in;
    wire        memb_write_enable;
    wire [1:0]  memb_write_line;
    wire [1:0]  memb_write_elem;
    wire [2:0]  memb_read_enable;
    wire [11:0] memb_read_sel;

    wire [47:0]  array_a_in;
    wire [14:0]  array_b_in;
    wire [125:0] array_data_out;

    control control_unit (
        .clk                (clk),
        .rst_n              (rst_n),
        .instruction        (instruction),
        .array_write_enable (array_write_enable),
        .array_clear        (array_clear),
        .dense_mode         (dense_mode),
        .store_row          (store_row),
        .store_col          (store_col),
        .store_byte_sel     (store_byte_sel),
        .mema_data_in       (mema_data_in),
        .mema_write_enable  (mema_write_enable),
        .mema_write_line    (mema_write_line),
        .mema_write_elem    (mema_write_elem),
        .mema_read_enable   (mema_read_enable),
        .mema_read_sel      (mema_read_sel),
        .memb_data_in       (memb_data_in),
        .memb_write_enable  (memb_write_enable),
        .memb_write_line    (memb_write_line),
        .memb_write_elem    (memb_write_elem),
        .memb_read_enable   (memb_read_enable),
        .memb_read_sel      (memb_read_sel),
        .ready_to_send      (ready_to_send)
    );

    memory_a memory_act (
        .clk          (clk),
        .write_enable (mema_write_enable),
        .write_line   (mema_write_line),
        .write_elem   (mema_write_elem),
        .data_in      (mema_data_in),
        .read_enable  (mema_read_enable),
        .read_sel     (mema_read_sel),
        .data_out     (array_a_in)
    );

    memory_b memory_wgt (
        .clk          (clk),
        .write_enable (memb_write_enable),
        .write_line   (memb_write_line),
        .write_elem   (memb_write_elem),
        .data_in      (memb_data_in),
        .read_enable  (memb_read_enable),
        .read_sel     (memb_read_sel),
        .data_out     (array_b_in)
    );

    array array_inst (
        .clk      (clk),
        .rst_n    (rst_n),
        .we       (array_write_enable),
        .clr      (array_clear),
        .dense    (dense_mode),
        .a_in     (array_a_in),
        .b_in     (array_b_in),
        .data_out (array_data_out)
    );

    // ------------------------------------------------------------------
    // Result readout: STORE latches {row, col, byte_sel}; the selected
    // accumulator byte drives `result` until the next STORE.
    // ------------------------------------------------------------------
    wire [13:0] acc [0:8];
    genvar i;
    generate
        for (i = 0; i < 9; i = i + 1) begin : extract_results
            assign acc[i] = array_data_out[14*i +: 14];
        end
    endgenerate

    // Index arithmetic in 4 bits — 2-bit operands would wrap modulo 4
    wire [3:0] sel_idx = {2'b0, store_row} * 4'd3 + {2'b0, store_col};
    wire [13:0] selected = (store_row < 2'd3 && store_col < 2'd3)
        ? acc[sel_idx]
        : 14'd0;

    assign result = store_byte_sel ? {2'b0, selected[13:8]}
                                   : selected[7:0];

endmodule

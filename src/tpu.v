/*
 * 1:2-sparse E2M1 x INT8 mini-TPU core: control + operand memories +
 * ONE time-multiplexed PE + result memory + result readout mux.
 *
 * Computes C = A x W with A a 3x8 INT8 activation matrix and W a
 * dense-equivalent 8x3 E2M1 weight matrix stored as 12 five-bit codes
 * {select, e2m1} (1:2 structured sparsity along the contraction
 * axis). Results are exact 14-bit signed values in the x2 integer
 * domain, read out one byte at a time via STORE.
 *
 * A single PE computes all 9 outputs sequentially (see control.v) --
 * the same datapath as the original 9-PE parallel array, reused nine
 * times per RUN. This is free: RUN's ~45-cycle latency is still a
 * small fraction of the ~1500 SPI clocks a RUN's operands take to
 * load, and removes the multi-port memory reads and per-PE
 * accumulator registers a spatial 3x3 array required.
 */

`default_nettype none

module tpu (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [15:0] instruction,
    output wire        ready_to_send,
    output wire [7:0]  result
);

    wire        pe_we;
    wire        pe_clr;
    wire        dense_mode;
    wire [1:0]  store_row, store_col;
    wire        store_byte_sel;

    wire [7:0]  mema_data_in;
    wire        mema_write_enable;
    wire [1:0]  mema_write_line;
    wire [2:0]  mema_write_elem;
    wire        mema_read_enable;
    wire [1:0]  mema_read_row;
    wire [1:0]  mema_read_pair;

    wire [4:0]  memb_data_in;
    wire        memb_write_enable;
    wire [1:0]  memb_write_line;
    wire [1:0]  memb_write_elem;
    wire        memb_read_enable;
    wire [1:0]  memb_read_col;
    wire [1:0]  memb_read_pair;

    wire        result_write_enable;
    wire [1:0]  result_write_row;
    wire [1:0]  result_write_col;

    wire [15:0] pe_a_in;
    wire [4:0]  pe_b_in;
    wire signed [13:0] pe_c_out;
    wire signed [13:0] selected;

    control control_unit (
        .clk                  (clk),
        .rst_n                (rst_n),
        .instruction          (instruction),
        .pe_we                (pe_we),
        .pe_clr               (pe_clr),
        .dense_mode           (dense_mode),
        .store_row            (store_row),
        .store_col            (store_col),
        .store_byte_sel       (store_byte_sel),
        .mema_data_in         (mema_data_in),
        .mema_write_enable    (mema_write_enable),
        .mema_write_line      (mema_write_line),
        .mema_write_elem      (mema_write_elem),
        .mema_read_enable     (mema_read_enable),
        .mema_read_row        (mema_read_row),
        .mema_read_pair       (mema_read_pair),
        .memb_data_in         (memb_data_in),
        .memb_write_enable    (memb_write_enable),
        .memb_write_line      (memb_write_line),
        .memb_write_elem      (memb_write_elem),
        .memb_read_enable     (memb_read_enable),
        .memb_read_col        (memb_read_col),
        .memb_read_pair       (memb_read_pair),
        .result_write_enable  (result_write_enable),
        .result_write_row     (result_write_row),
        .result_write_col     (result_write_col),
        .ready_to_send        (ready_to_send)
    );

    memory_a memory_act (
        .clk          (clk),
        .write_enable (mema_write_enable),
        .write_line   (mema_write_line),
        .write_elem   (mema_write_elem),
        .data_in      (mema_data_in),
        .read_enable  (mema_read_enable),
        .read_row     (mema_read_row),
        .read_pair    (mema_read_pair),
        .data_out     (pe_a_in)
    );

    memory_b memory_wgt (
        .clk          (clk),
        .write_enable (memb_write_enable),
        .write_line   (memb_write_line),
        .write_elem   (memb_write_elem),
        .data_in      (memb_data_in),
        .read_enable  (memb_read_enable),
        .read_col     (memb_read_col),
        .read_pair    (memb_read_pair),
        .data_out     (pe_b_in)
    );

    // One PE, time-multiplexed across all 9 outputs. a_out/b_out are
    // systolic pass-through ports for neighboring PEs -- unused here
    // (no neighbors), left unconnected so synthesis drops a_reg/b_reg.
    pe pe_inst (
        .clk    (clk),
        .rst_n  (rst_n),
        .we     (pe_we),
        .clr    (pe_clr),
        .dense  (dense_mode),
        .a_in   (pe_a_in),
        .b_in   (pe_b_in),
        .a_out  (),
        .b_out  (),
        .c_out  (pe_c_out)
    );

    result_mem result_store (
        .clk          (clk),
        .rst_n        (rst_n),
        .write_enable (result_write_enable),
        .write_row    (result_write_row),
        .write_col    (result_write_col),
        .write_data   (pe_c_out),
        .read_row     (store_row),
        .read_col     (store_col),
        .read_data    (selected)
    );

    assign result = store_byte_sel ? {2'b0, selected[13:8]}
                                   : selected[7:0];

endmodule

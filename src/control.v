/*
 * Control unit -- time-multiplexed single-PE sequencer.
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
 *          Sequences all 9 (row,col) outputs through ONE physical PE:
 *          for each output, 4 accumulate cycles (one per weight-code
 *          slot, matching the original per-RUN K=8 sparse / K=4
 *          dense contraction depth exactly -- same PE, same
 *          datapath, same accumulator width, just one output at a
 *          time instead of nine in parallel) followed by a commit
 *          cycle that latches the finished accumulator into
 *          result_mem and clears the PE for the next output. Total
     *          latency ~45 cycles -- about 1.3% of the >=3456 clk cycles
     *          needed to transfer a RUN's 36 operand instructions at
     *          SCLK <= clk/6, so the
 *          9x serialization is free in practice (this is why: with
 *          no skewed wavefront to feed, a single read port per
 *          memory replaces the old 3-concurrent-port design).
 *
 *   STORE: [13]    byte_sel (0 = acc[7:0], 1 = {2'b0, acc[13:8]})
 *          [12:11] row, [10:9] col
 *          Latches the selection; result output holds until next STORE.
 *          Reads result_mem, which (like the old per-PE accumulators)
 *          holds every output until the next RUN overwrites it.
 */

`default_nettype none

module control (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] instruction,

    output wire        pe_we,       // accumulate this cycle
    output wire        pe_clr,      // clear PE accumulator (new output starting)
    output reg         dense_mode,
    output reg  [1:0]  store_row,
    output reg  [1:0]  store_col,
    output reg         store_byte_sel,

    output wire [7:0]  mema_data_in,
    output wire        mema_write_enable,
    output wire [1:0]  mema_write_line,
    output wire [2:0]  mema_write_elem,
    output wire        mema_read_enable,
    output wire [1:0]  mema_read_row,
    output wire [1:0]  mema_read_pair,

    output wire [4:0]  memb_data_in,
    output wire        memb_write_enable,
    output wire [1:0]  memb_write_line,
    output wire [1:0]  memb_write_elem,
    output wire        memb_read_enable,
    output wire [1:0]  memb_read_col,
    output wire [1:0]  memb_read_pair,

    output wire        result_write_enable,
    output wire [1:0]  result_write_row,
    output wire [1:0]  result_write_col,

    output reg         ready_to_send
);

    localparam [1:0] RUN   = 2'b01;
    localparam [1:0] LOAD  = 2'b10;
    localparam [1:0] STORE = 2'b11;

    wire [1:0] opcode     = instruction[15:14];
    wire       mem_select = instruction[13];
    wire [1:0] line       = instruction[12:11];
    wire [2:0] elem       = instruction[10:8];
    wire [7:0] imm        = instruction[7:0];

    wire is_load  = (opcode == LOAD);
    wire is_store = (opcode == STORE);

    // ------------------------------------------------------------------
    // Sequencer: (cur_row, cur_col) walk the 9 outputs row-major; step
    // walks the 4 weight-code slots of the current output. `running`
    // gates the 4 accumulate cycles; `commit` is the 1-cycle pulse
    // that latches the finished output and advances to the next (or
    // signals done on the last one).
    // ------------------------------------------------------------------
    reg [1:0] cur_row, cur_col;
    reg [1:0] step;
    reg       running;
    reg       commit;

    wire is_run_issue = (opcode == RUN) && !running && !commit;
    wire last_output   = (cur_row == 2'd2) && (cur_col == 2'd2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            dense_mode <= 1'b0;
        else if (is_run_issue)
            dense_mode <= instruction[13];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cur_row       <= 2'd0;
            cur_col       <= 2'd0;
            step          <= 2'd0;
            running       <= 1'b0;
            commit        <= 1'b0;
            ready_to_send <= 1'b0;
        end else begin
            ready_to_send <= 1'b0;
            if (is_run_issue) begin
                cur_row <= 2'd0;
                cur_col <= 2'd0;
                step    <= 2'd0;
                running <= 1'b1;
                commit  <= 1'b0;
            end else if (running) begin
                if (step == 2'd3) begin
                    running <= 1'b0;
                    commit  <= 1'b1;
                end else begin
                    step <= step + 2'd1;
                end
            end else if (commit) begin
                commit <= 1'b0;
                if (last_output) begin
                    ready_to_send <= 1'b1;
                end else begin
                    if (cur_col == 2'd2) begin
                        cur_row <= cur_row + 2'd1;
                        cur_col <= 2'd0;
                    end else begin
                        cur_col <= cur_col + 2'd1;
                    end
                    step    <= 2'd0;
                    running <= 1'b1;
                end
            end
        end
    end

    // Memory reads: single port each, addressed by the current output
    // and weight-code step -- valid throughout `running`.
    assign mema_read_enable = running;
    assign mema_read_row    = cur_row;
    assign mema_read_pair   = step;

    assign memb_read_enable = running;
    assign memb_read_col    = cur_col;
    assign memb_read_pair   = step;

    // PE control: accumulate while running; clear one cycle before
    // the first accumulate of an output (issue, or the commit cycle
    // that precedes the next output) -- harmless on the very last
    // commit, since the next RUN clears again anyway.
    assign pe_we  = running;
    assign pe_clr = is_run_issue || commit;

    // Result memory: latch the finished accumulator on the commit cycle.
    assign result_write_enable = commit;
    assign result_write_row    = cur_row;
    assign result_write_col    = cur_col;

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

endmodule

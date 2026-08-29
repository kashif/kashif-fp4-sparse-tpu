`timescale 1ns/1ps
`default_nettype none

module control_busy_tb;
    reg         clk = 1'b0;
    reg         rst_n = 1'b0;
    reg  [15:0] instruction = 16'd0;
    wire        mema_write_enable;
    wire        memb_write_enable;
    wire [1:0]  store_row;
    wire [1:0]  store_col;
    wire        store_byte_sel;
    wire        ready_to_send;

    always #5 clk = ~clk;

    control dut (
        .clk               (clk),
        .rst_n             (rst_n),
        .instruction       (instruction),
        .store_row         (store_row),
        .store_col         (store_col),
        .store_byte_sel    (store_byte_sel),
        .mema_write_enable (mema_write_enable),
        .memb_write_enable (memb_write_enable),
        .ready_to_send     (ready_to_send)
    );

    task present(input [15:0] word);
        begin
            @(negedge clk);
            instruction = word;
            @(negedge clk);
            instruction = 16'd0;
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        // Sparse RUN starts the engine.
        present(16'h4000);
        if (!dut.busy || dut.dense_mode || ready_to_send)
            $fatal(1, "RUN did not enter the expected busy state");

        // A, B, STORE, and another RUN must have no effect while busy.
        @(negedge clk);
        instruction = 16'h8001; // LOAD A
        #1;
        if (mema_write_enable || memb_write_enable)
            $fatal(1, "LOAD A accepted while busy");
        @(negedge clk);
        instruction = 16'ha001; // LOAD B
        #1;
        if (mema_write_enable || memb_write_enable)
            $fatal(1, "LOAD B accepted while busy");
        present(16'hea00);       // STORE high byte, row=1, col=1
        if (store_row != 0 || store_col != 0 || store_byte_sel != 0)
            $fatal(1, "STORE selection changed while busy");
        present(16'h6000);       // dense RUN
        if (dut.dense_mode)
            $fatal(1, "second RUN changed mode while busy");

        // Completion is sticky until a new RUN is accepted.
        wait (ready_to_send === 1'b1);
        repeat (5) @(posedge clk);
        if (!ready_to_send)
            $fatal(1, "ready was not sticky");

        // Commands are accepted again when idle.
        @(negedge clk);
        instruction = 16'h8001;
        #1;
        if (!mema_write_enable)
            $fatal(1, "LOAD A not accepted while idle");
        @(negedge clk);
        instruction = 16'd0;

        present(16'h6000);
        if (!dut.busy || !dut.dense_mode || ready_to_send)
            $fatal(1, "new RUN did not clear ready and latch dense mode");

        $display("PASS: controller busy gating and sticky ready");
        $finish;
    end
endmodule

`default_nettype wire

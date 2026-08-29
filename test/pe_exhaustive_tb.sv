`timescale 1ns/1ps
`default_nettype none

module pe_exhaustive_tb;
    reg         clk = 1'b0;
    reg         rst_n = 1'b0;
    reg         we = 1'b0;
    reg         clr = 1'b0;
    reg         dense = 1'b0;
    reg  [15:0] a_in = 16'd0;
    reg  [4:0]  b_in = 5'd0;
    wire [13:0] c_out;

    integer act_i;
    integer nibble_i;
    integer select_i;
    integer mode_i;
    integer mag_i;
    integer weight_i;
    integer expected_i;
    integer checks;

    always #5 clk = ~clk;

    pe dut (
        .clk   (clk),
        .rst_n (rst_n),
        .we    (we),
        .clr   (clr),
        .dense (dense),
        .a_in  (a_in),
        .b_in  (b_in),
        .a_out (),
        .b_out (),
        .c_out (c_out)
    );

    function integer decode_mag(input [2:0] code);
        begin
            case (code)
                3'd0: decode_mag = 0;
                3'd1: decode_mag = 1;
                3'd2: decode_mag = 2;
                3'd3: decode_mag = 3;
                3'd4: decode_mag = 4;
                3'd5: decode_mag = 6;
                3'd6: decode_mag = 8;
                3'd7: decode_mag = 12;
            endcase
        end
    endfunction

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        checks = 0;

        for (mode_i = 0; mode_i < 2; mode_i = mode_i + 1) begin
            for (select_i = 0; select_i < 2; select_i = select_i + 1) begin
                for (nibble_i = 0; nibble_i < 16; nibble_i = nibble_i + 1) begin
                    mag_i = decode_mag(nibble_i[2:0]);
                    weight_i = nibble_i[3] ? -mag_i : mag_i;
                    for (act_i = -128; act_i < 128; act_i = act_i + 1) begin
                        // Dense mode and sparse select=0 consume the even
                        // byte. Sparse select=1 consumes the odd byte. Fill
                        // the unselected byte with a distinct value.
                        if (!mode_i && select_i)
                            a_in = {act_i[7:0], 8'h55};
                        else
                            a_in = {8'haa, act_i[7:0]};
                        b_in = {select_i[0], nibble_i[3:0]};
                        dense = mode_i[0];
                        expected_i = act_i * weight_i;

                        @(negedge clk);
                        clr = 1'b1;
                        we  = 1'b0;
                        @(negedge clk);
                        clr = 1'b0;
                        we  = 1'b1;
                        @(posedge clk);
                        #1;
                        if ($signed(c_out) !== expected_i) begin
                            $display("FAIL mode=%0d sel=%0d nibble=%0h act=%0d got=%0d expected=%0d",
                                     mode_i, select_i, nibble_i, act_i,
                                     $signed(c_out), expected_i);
                            $fatal(1);
                        end
                        we = 1'b0;
                        checks = checks + 1;
                    end
                end
            end
        end

        $display("PASS: %0d exhaustive PE products", checks);
        $finish;
    end
endmodule

`default_nettype wire

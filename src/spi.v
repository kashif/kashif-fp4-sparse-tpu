/*
 * Single-clock SPI instruction receiver.
 *
 * MOSI shifts in LSB-first on external SCLK rising edges while CS is low.
 * SCLK, CS, and MOSI are first synchronized into the project `clk` domain;
 * all state below is clocked only by `clk`. This avoids using a general-
 * purpose input as an internal clock and removes the SPI-to-core CDC.
 *
 * SCLK must be <= clk/6, with MOSI stable before each rising edge. The
 * receiver accepts exactly 16 bits per CS assertion. Raising CS discards a
 * partial frame; after a complete word, further SCLK edges are ignored until
 * CS has been observed high. A completed instruction is presented for one
 * `clk` cycle, with zero (NOP) at all other times.
 */

`default_nettype none

module spi (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        mosi,
    input  wire        cs,
    input  wire        sclk,

    output wire [15:0] data_buffer_output
);

    // Synchronize all asynchronous interface pins. Keeping the three SCLK
    // and CS stages together helps implementation tools place them as CDC
    // synchronizers. MOSI uses the same depth so its sampled value is aligned
    // with the delayed SCLK edge indication.
    (* async_reg = "true" *) reg [2:0] sclk_sync;
    (* async_reg = "true" *) reg [2:0] cs_sync;
    (* async_reg = "true" *) reg [2:0] mosi_sync;

    wire sclk_rising = (sclk_sync[2:1] == 2'b01);
    wire cs_active   = !cs_sync[2];

    reg [14:0] shift_reg;
    reg [3:0]  bit_counter;
    reg        frame_complete;
    reg [15:0] completed_word;
    reg        data_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sclk_sync <= 3'b000;
            cs_sync   <= 3'b111;
            mosi_sync <= 3'b000;
        end else begin
            sclk_sync <= {sclk_sync[1:0], sclk};
            cs_sync   <= {cs_sync[1:0], cs};
            mosi_sync <= {mosi_sync[1:0], mosi};
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_reg      <= 15'd0;
            bit_counter    <= 4'd0;
            frame_complete <= 1'b0;
            completed_word <= 16'd0;
            data_ready     <= 1'b0;
        end else begin
            data_ready <= 1'b0;

            if (!cs_active) begin
                // CS high terminates the current transaction. A partial word
                // is discarded; a completed word remains available only via
                // its one-cycle data_ready pulse.
                shift_reg      <= 15'd0;
                bit_counter    <= 4'd0;
                frame_complete <= 1'b0;
            end else if (sclk_rising && !frame_complete) begin
                shift_reg <= {mosi_sync[2], shift_reg[14:1]};
                if (bit_counter == 4'd15) begin
                    completed_word <= {mosi_sync[2], shift_reg[14:0]};
                    bit_counter    <= 4'd0;
                    frame_complete <= 1'b1;
                    data_ready     <= 1'b1;
                end else begin
                    bit_counter <= bit_counter + 4'd1;
                end
            end
        end
    end

    assign data_buffer_output = data_ready ? completed_word : 16'd0;

endmodule

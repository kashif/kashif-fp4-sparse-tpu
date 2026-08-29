/*
 * SPI slave — ported from the reference mini-TPU spi.v, widened to
 * 16-bit instructions. Receive-only: all results are read via STORE
 * on uo_out, so the reference's MISO readback stream is omitted
 * (saves the readback mux and counter).
 *
 * MOSI shifts in on posedge SCLK while CS is low, LSB-first (bit 0 of
 * the instruction is sent first). When the 16th bit lands, the
 * completed word is latched in the SCLK domain and a one-bit toggle is
 * passed through a two-flop synchronizer.  The clk domain detects the
 * toggle and presents the already-stable word to the control unit for
 * exactly one clk cycle (0 = NOP otherwise).
 *
 * SCLK must be <= clk/6.  Besides matching the external interface
 * timing, this leaves ample time for the completed-word bus to settle
 * before the synchronized toggle is observed in the clk domain.
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

    reg [15:0] data_buffer;
    reg [3:0]  bit_counter;

    // The word is held stable from completion until the next complete
    // transaction.  Only the one-bit toggle passes through synchronizer
    // flops; by the time it arrives, completed_word has been stable for
    // at least two clk edges.
    reg [15:0] completed_word;
    reg        completion_toggle;
    (* async_reg = "true" *) reg toggle_meta;
    (* async_reg = "true" *) reg toggle_sync;
    reg        toggle_seen;
    reg        data_ready;

    // CS deassertion asynchronously discards a partial frame.  This makes
    // abort/restart behavior independent of whether SCLK continues while
    // CS is high and prevents an incomplete word from becoming valid.
    always @(posedge sclk or posedge cs or negedge rst_n) begin
        if (!rst_n) begin
            data_buffer <= 16'd0;
            bit_counter <= 4'd0;
        end else if (cs) begin
            data_buffer <= 16'd0;
            bit_counter <= 4'd0;
        end else begin
            data_buffer <= {mosi, data_buffer[15:1]};
            if (bit_counter == 4'd15)
                bit_counter <= 4'd0;
            else
                bit_counter <= bit_counter + 4'd1;
        end
    end

    always @(posedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            completed_word    <= 16'd0;
            completion_toggle <= 1'b0;
        end else if (!cs && bit_counter == 4'd15) begin
            // Include the bit sampled on this edge; data_buffer itself is
            // updated by nonblocking assignment in the other SCLK block.
            completed_word    <= {mosi, data_buffer[15:1]};
            completion_toggle <= ~completion_toggle;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            toggle_meta      <= 1'b0;
            toggle_sync      <= 1'b0;
            toggle_seen      <= 1'b0;
            data_ready       <= 1'b0;
        end else begin
            toggle_meta <= completion_toggle;
            toggle_sync <= toggle_meta;

            data_ready <= 1'b0;
            if (toggle_sync != toggle_seen) begin
                toggle_seen <= toggle_sync;
                data_ready  <= 1'b1;
            end
        end
    end

    assign data_buffer_output = data_ready ? completed_word : 16'd0;

endmodule

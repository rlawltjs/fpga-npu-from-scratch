// ---------------------------------------------------------------------------
// maxpool2x2.v
//
// 2x2 / stride-2 max-pooling engine. Channel count and input feature-map
// size are runtime inputs so a single instance is reused for LeNet's S2
// (6 ch, 28x28 -> 14x14) and S4 (16 ch, 10x10 -> 5x5) layers.
//
// For each output pixel, reads the 4 contributing input pixels one at a
// time (memory has 1-cycle read latency) and writes out their max.
// ---------------------------------------------------------------------------
`timescale 1ns/1ps

module maxpool2x2 #(
    parameter DATA_W = 8,
    parameter ADDR_W = 16
)(
    input                       clk,
    input                       rst_n,
    input                       start,
    output reg                  done,

    input      [5:0]            channels,
    input      [6:0]            in_size,     // must be even

    output reg [ADDR_W-1:0]     ifmap_addr,
    input      signed [DATA_W-1:0] ifmap_dout,

    output reg [ADDR_W-1:0]     ofmap_addr,
    output reg signed [DATA_W-1:0] ofmap_din,
    output reg                  ofmap_we
);

    // See conv2d_engine.v note: bram_dp read latency is 2 cycles (address
    // register + memory output register), so each of the 4 window reads
    // needs its own ADDR -> WAIT -> (capture) sequence.
    localparam S_IDLE = 4'd0,
               S_A0    = 4'd1,   // drive addr for top-left
               S_W0    = 4'd2,   // wait for memory
               S_A1    = 4'd3,   // capture top-left,  drive addr top-right
               S_W1    = 4'd4,
               S_A2    = 4'd5,   // capture top-right, drive addr bottom-left
               S_W2    = 4'd6,
               S_A3    = 4'd7,   // capture bottom-left, drive addr bottom-right
               S_W3    = 4'd8,
               S_WRITE = 4'd9,   // capture bottom-right, compute max, write
               S_NEXT  = 4'd10,
               S_DONE  = 4'd11;

    reg [3:0]  state;
    reg [6:0]  out_size;
    reg [5:0]  ch;
    reg [6:0]  oy, ox;
    reg signed [DATA_W-1:0] v0, v1, v2;

    function [ADDR_W-1:0] addr_f;
        input [15:0] c, r, cc, sz;
        begin
            addr_f = c * (sz * sz) + r * sz + cc;
        end
    endfunction

    function signed [DATA_W-1:0] max4;
        input signed [DATA_W-1:0] a, b, c, d;
        reg   signed [DATA_W-1:0] m01, m23;
        begin
            m01  = (a > b) ? a : b;
            m23  = (c > d) ? c : d;
            max4 = (m01 > m23) ? m01 : m23;
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done  <= 1'b0;
            ofmap_we <= 1'b0;
        end else begin
            ofmap_we <= 1'b0;
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        out_size <= in_size >> 1;
                        ch <= 6'd0; oy <= 7'd0; ox <= 7'd0;
                        state <= S_A0;
                    end
                end

                S_A0: begin
                    ifmap_addr <= addr_f(ch, oy*2,   ox*2,   in_size);
                    state <= S_W0;
                end
                S_W0: state <= S_A1;
                S_A1: begin
                    v0 <= ifmap_dout;
                    ifmap_addr <= addr_f(ch, oy*2,   ox*2+1, in_size);
                    state <= S_W1;
                end
                S_W1: state <= S_A2;
                S_A2: begin
                    v1 <= ifmap_dout;
                    ifmap_addr <= addr_f(ch, oy*2+1, ox*2,   in_size);
                    state <= S_W2;
                end
                S_W2: state <= S_A3;
                S_A3: begin
                    v2 <= ifmap_dout;
                    ifmap_addr <= addr_f(ch, oy*2+1, ox*2+1, in_size);
                    state <= S_W3;
                end
                S_W3: state <= S_WRITE;
                S_WRITE: begin
                    ofmap_addr <= ch * (out_size*out_size) + oy*out_size + ox;
                    ofmap_din  <= max4(v0, v1, v2, ifmap_dout);
                    ofmap_we   <= 1'b1;
                    state <= S_NEXT;
                end

                S_NEXT: begin
                    if (ox == out_size-1) begin
                        ox <= 7'd0;
                        if (oy == out_size-1) begin
                            oy <= 7'd0;
                            if (ch == channels-1)
                                state <= S_DONE;
                            else begin
                                ch <= ch + 6'd1;
                                state <= S_A0;
                            end
                        end else begin
                            oy <= oy + 7'd1;
                            state <= S_A0;
                        end
                    end else begin
                        ox <= ox + 7'd1;
                        state <= S_A0;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    if (!start) state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

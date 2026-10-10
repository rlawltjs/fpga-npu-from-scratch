// ---------------------------------------------------------------------------
// conv2d_engine.v
//
// Sequential (MAC-per-2-cycles) valid-convolution engine, K x K kernel,
// stride 1, no padding. Channel count and feature-map size are RUNTIME
// inputs (not compile-time parameters) so that a single instance of this
// engine is reused for both LeNet conv layers (C1: 1->6 ch, 32x32 in;
// C3: 6->16 ch, 14x14 in) under control of the top-level FSM.
//
// This is a deliberately simple / non-pipelined "Stage 1" baseline
// (2 clock cycles per MAC term: 1 to drive the address, 1 to capture data
// and accumulate). It establishes the memory-map, control-FSM, and
// quantization conventions that later stages (MobileNet depthwise conv,
// YOLO CSP blocks) will build on and optimize (pipelining, parallel MACs).
//
// Data layout in memory (row-major, channel-major):
//   ifmap[ch][row][col]        addr = ch*(size*size) + row*size + col
//   weight[oc][ic][ky][kx]     addr = ((oc*in_ch + ic)*K + ky)*K + kx
//   bias[oc]                   addr = oc
//   ofmap[oc][oy][ox]          addr = oc*(out_size*out_size) + oy*out_size + ox
// ---------------------------------------------------------------------------
`timescale 1ns/1ps

module conv2d_engine #(
    parameter DATA_W  = 8,     // activation / weight bit width (signed)
    parameter ACC_W   = 32,    // accumulator bit width (signed)
    parameter K       = 5,     // kernel size (fixed 5x5 for LeNet C1/C3)
    parameter ADDR_W  = 16,    // shared memory address bus width
    parameter SHIFT   = 0      // requantization right-shift amount
)(
    input                       clk,
    input                       rst_n,
    input                       start,
    output reg                  done,

    input                       apply_relu,   // 1 = ReLU after bias-add
    input      [5:0]            in_ch,        // e.g. 1 or 6
    input      [5:0]            out_ch,       // e.g. 6 or 16
    input      [6:0]            in_size,      // e.g. 32 or 14

    // ifmap read port
    output reg [ADDR_W-1:0]     ifmap_addr,
    input      signed [DATA_W-1:0] ifmap_dout,

    // weight read port
    output reg [ADDR_W-1:0]     weight_addr,
    input      signed [DATA_W-1:0] weight_dout,

    // bias read port
    output reg [ADDR_W-1:0]     bias_addr,
    input      signed [ACC_W-1:0] bias_dout,

    // ofmap write port
    output reg [ADDR_W-1:0]     ofmap_addr,
    output reg signed [DATA_W-1:0] ofmap_din,
    output reg                  ofmap_we
);

    localparam S_IDLE      = 4'd0,
               S_ADDR       = 4'd1,
               S_READ       = 4'd2,
               S_BIAS_ADDR  = 4'd3,
               S_BIAS_READ  = 4'd4,
               S_WRITE      = 4'd5,
               S_NEXT       = 4'd6,
               S_DONE       = 4'd7;

    reg [3:0]  state;
    reg [6:0]  out_size;
    reg [5:0]  oc, ic;
    reg [6:0]  oy, ox, ky, kx;
    reg signed [ACC_W-1:0] acc;

    `include "requant_func.vh"

    // -------- combinational address helpers --------
    function [ADDR_W-1:0] ifmap_addr_f;
        input [15:0] ch, row, col, sz;
        begin
            ifmap_addr_f = ch * (sz * sz) + row * sz + col;
        end
    endfunction

    function [ADDR_W-1:0] weight_addr_f;
        input [15:0] o_ch, i_ch, ky_, kx_, nic;
        begin
            weight_addr_f = ((o_ch * nic + i_ch) * K + ky_) * K + kx_;
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            ofmap_we <= 1'b0;
        end else begin
            ofmap_we <= 1'b0;  // default; asserted explicitly in S_WRITE
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        out_size <= in_size - K + 1;
                        oc <= 6'd0; oy <= 7'd0; ox <= 7'd0;
                        ic <= 6'd0; ky <= 7'd0; kx <= 7'd0;
                        acc <= {ACC_W{1'b0}};
                        state <= S_ADDR;
                    end
                end

                S_ADDR: begin
                    ifmap_addr  <= ifmap_addr_f(ic, oy + ky, ox + kx, in_size);
                    weight_addr <= weight_addr_f(oc, ic, ky, kx, in_ch);
                    state <= S_READ;
                end

                S_READ: begin
                    acc <= acc + ifmap_dout * weight_dout;
                    if (kx == K-1) begin
                        kx <= 7'd0;
                        if (ky == K-1) begin
                            ky <= 7'd0;
                            if (ic == in_ch-1)
                                state <= S_BIAS_ADDR;      // inner MAC loop done
                            else begin
                                ic <= ic + 6'd1;
                                state <= S_ADDR;
                            end
                        end else begin
                            ky <= ky + 7'd1;
                            state <= S_ADDR;
                        end
                    end else begin
                        kx <= kx + 7'd1;
                        state <= S_ADDR;
                    end
                end

                S_BIAS_ADDR: begin
                    bias_addr <= {10'd0, oc};
                    state <= S_BIAS_READ;
                end

                S_BIAS_READ: begin
                    acc <= acc + bias_dout;
                    state <= S_WRITE;
                end

                S_WRITE: begin
                    ofmap_addr <= oc * (out_size * out_size) + oy * out_size + ox;
                    ofmap_din  <= requant_relu(acc, apply_relu);
                    ofmap_we   <= 1'b1;
                    state <= S_NEXT;
                end

                S_NEXT: begin
                    ic <= 6'd0; ky <= 7'd0; kx <= 7'd0;
                    acc <= {ACC_W{1'b0}};
                    if (ox == out_size-1) begin
                        ox <= 7'd0;
                        if (oy == out_size-1) begin
                            oy <= 7'd0;
                            if (oc == out_ch-1)
                                state <= S_DONE;
                            else begin
                                oc <= oc + 6'd1;
                                state <= S_ADDR;
                            end
                        end else begin
                            oy <= oy + 7'd1;
                            state <= S_ADDR;
                        end
                    end else begin
                        ox <= ox + 7'd1;
                        state <= S_ADDR;
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

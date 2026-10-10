// ---------------------------------------------------------------------------
// fc_engine.v
//
// Sequential fully-connected (matrix-vector multiply) engine. Input/output
// dimensions are runtime inputs so a single instance is reused for LeNet's
// F5 (400->120), F6 (120->84), and OUTPUT (84->10) layers. apply_relu is
// also a runtime input since F5/F6 use ReLU but the final classification
// layer must output raw scores for argmax.
//
// weight layout: weight[o][i]  addr = o*in_dim + i
// bias   layout: bias[o]       addr = o
// ---------------------------------------------------------------------------
`timescale 1ns/1ps

module fc_engine #(
    parameter DATA_W = 8,
    parameter ACC_W  = 32,
    parameter ADDR_W = 16,
    parameter SHIFT  = 0
)(
    input                       clk,
    input                       rst_n,
    input                       start,
    output reg                  done,

    input                       apply_relu,
    input      [9:0]            in_dim,     // e.g. 400 / 120 / 84
    input      [8:0]            out_dim,    // e.g. 120 / 84 / 10

    output reg [ADDR_W-1:0]     ivec_addr,
    input      signed [DATA_W-1:0] ivec_dout,

    output reg [ADDR_W-1:0]     weight_addr,
    input      signed [DATA_W-1:0] weight_dout,

    output reg [ADDR_W-1:0]     bias_addr,
    input      signed [ACC_W-1:0] bias_dout,

    output reg [ADDR_W-1:0]     ovec_addr,
    output reg signed [DATA_W-1:0] ovec_din,
    output reg                  ovec_we
);

    // See conv2d_engine.v note: bram_dp read latency is 2 cycles
    // (address register + memory output register), hence the *_WAIT states.
    localparam S_IDLE     = 4'd0,
               S_ADDR      = 4'd1,
               S_WAIT      = 4'd2,
               S_READ      = 4'd3,
               S_BIAS_ADDR = 4'd4,
               S_BIAS_WAIT = 4'd5,
               S_BIAS_READ = 4'd6,
               S_WRITE     = 4'd7,
               S_NEXT      = 4'd8,
               S_DONE      = 4'd9;

    reg [3:0]  state;
    reg [8:0]  o;
    reg [9:0]  i;
    reg signed [ACC_W-1:0] acc;

    `include "requant_func.vh"

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            done    <= 1'b0;
            ovec_we <= 1'b0;
        end else begin
            ovec_we <= 1'b0;
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        o <= 9'd0; i <= 10'd0;
                        acc <= {ACC_W{1'b0}};
                        state <= S_ADDR;
                    end
                end

                S_ADDR: begin
                    ivec_addr   <= i;
                    weight_addr <= o * in_dim + i;
                    state <= S_WAIT;
                end

                S_WAIT: begin
                    state <= S_READ;
                end

                S_READ: begin
                    acc <= acc + ivec_dout * weight_dout;
                    if (i == in_dim-1)
                        state <= S_BIAS_ADDR;
                    else begin
                        i <= i + 10'd1;
                        state <= S_ADDR;
                    end
                end

                S_BIAS_ADDR: begin
                    bias_addr <= {7'd0, o};
                    state <= S_BIAS_WAIT;
                end

                S_BIAS_WAIT: begin
                    state <= S_BIAS_READ;
                end

                S_BIAS_READ: begin
                    acc <= acc + bias_dout;
                    state <= S_WRITE;
                end

                S_WRITE: begin
                    ovec_addr <= {7'd0, o};
                    ovec_din  <= requant_relu(acc, apply_relu);
                    ovec_we   <= 1'b1;
                    state <= S_NEXT;
                end

                S_NEXT: begin
                    i <= 10'd0;
                    acc <= {ACC_W{1'b0}};
                    if (o == out_dim-1)
                        state <= S_DONE;
                    else begin
                        o <= o + 9'd1;
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

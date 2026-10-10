// ---------------------------------------------------------------------------
// argmax_unit.v
//
// Scans n scalar values (the final FC layer's raw class scores) one at a
// time and returns the index of the maximum -> the predicted class.
// ---------------------------------------------------------------------------
`timescale 1ns/1ps

module argmax_unit #(
    parameter DATA_W = 8,
    parameter ADDR_W = 16
)(
    input                       clk,
    input                       rst_n,
    input                       start,
    output reg                  done,

    input      [7:0]            n,          // number of classes, e.g. 10

    output reg [ADDR_W-1:0]     ivec_addr,
    input      signed [DATA_W-1:0] ivec_dout,

    output reg [7:0]            max_idx,
    output reg signed [DATA_W-1:0] max_val
);

    localparam S_IDLE = 3'd0,
               S_ADDR  = 3'd1,
               S_READ  = 3'd2,
               S_DONE  = 3'd3;

    reg [2:0] state;
    reg [7:0] idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done  <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        idx <= 8'd0;
                        max_idx <= 8'd0;
                        max_val <= -128;
                        state <= S_ADDR;
                    end
                end

                S_ADDR: begin
                    ivec_addr <= {8'd0, idx};
                    state <= S_READ;
                end

                S_READ: begin
                    if (ivec_dout > max_val) begin
                        max_val <= ivec_dout;
                        max_idx <= idx;
                    end
                    if (idx == n-1)
                        state <= S_DONE;
                    else begin
                        idx <= idx + 8'd1;
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

// ---------------------------------------------------------------------------
// bram_dp.v
//
// Generic synchronous memory with one write port and one registered
// (1-cycle latency) read port. Used throughout the LeNet NPU for:
//   - input image storage
//   - ping-pong feature-map buffers between layers
//   - per-layer weight / bias ROMs (loaded via $readmemh at elaboration)
//
// On real FPGA toolchains (Vivado/Quartus) this behavioral description
// infers a single Block RAM per instance.
// ---------------------------------------------------------------------------
`timescale 1ns/1ps

module bram_dp #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH      = 1024,
    parameter ADDR_WIDTH = 16,
    parameter INIT_FILE  = ""      // path to $readmemh hex file, "" => zero init
)(
    input                                clk,

    input                                we,
    input      [ADDR_WIDTH-1:0]          waddr,
    input      signed [DATA_WIDTH-1:0]   din,

    input      [ADDR_WIDTH-1:0]          raddr,
    output reg signed [DATA_WIDTH-1:0]   dout
);

    reg signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    integer k;
    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end else begin
            for (k = 0; k < DEPTH; k = k + 1)
                mem[k] = {DATA_WIDTH{1'b0}};
        end
    end

    always @(posedge clk) begin
        if (we)
            mem[waddr] <= din;
        dout <= mem[raddr];
    end

endmodule

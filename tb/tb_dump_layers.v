// tb_dump_layers.v - dumps intermediate feature maps after every stage
// completes, for comparison against scripts/golden_model.py
`timescale 1ns/1ps

module tb_dump_layers;

    reg clk, rst_n, start;
    wire busy, class_valid;
    wire [7:0] class_idx;

    lenet_top #(
        .DATA_W(8), .ACC_W(32), .ADDR_W(16), .SHIFT(4),
        .WEIGHTS_DIR("/home/claude/lenet_npu/sim")
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .busy(busy), .class_valid(class_valid), .class_idx(class_idx)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    reg [3:0] prev_stage;

    always @(posedge clk) begin
        if (rst_n && dut.stage !== prev_stage) begin
            // dump the memory that just finished being written, at the moment
            // we transition AWAY from the stage that wrote it
            case (prev_stage)
                4'd1: $writememh("/home/claude/lenet_npu/sim/dbg_c1_rtl.hex", dut.u_fm_a.mem, 0, 4703); // CONV1 -> fm_a
                4'd2: $writememh("/home/claude/lenet_npu/sim/dbg_s2_rtl.hex", dut.u_fm_b.mem, 0, 4703); // POOL1 -> fm_b
                4'd3: $writememh("/home/claude/lenet_npu/sim/dbg_c3_rtl.hex", dut.u_fm_a.mem, 0, 4703); // CONV2 -> fm_a
                4'd4: $writememh("/home/claude/lenet_npu/sim/dbg_s4_rtl.hex", dut.u_fm_b.mem, 0, 4703); // POOL2 -> fm_b
                4'd5: $writememh("/home/claude/lenet_npu/sim/dbg_f5_rtl.hex", dut.u_fm_a.mem, 0, 4703); // FC1   -> fm_a
                4'd6: $writememh("/home/claude/lenet_npu/sim/dbg_f6_rtl.hex", dut.u_fm_b.mem, 0, 4703); // FC2   -> fm_b
                4'd7: $writememh("/home/claude/lenet_npu/sim/dbg_out_rtl.hex", dut.u_fm_a.mem, 0, 4703); // FC3   -> fm_a
                default: ;
            endcase
            $display("[t=%0t] stage %0d -> %0d", $time, prev_stage, dut.stage);
            prev_stage = dut.stage;
        end
    end

    initial begin
        prev_stage = 4'hF;
        rst_n = 0; start = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        start = 1;
        wait (class_valid == 1'b1);
        @(posedge clk);
        $display("Predicted class index : %0d", class_idx);
        start = 0;
        repeat (5) @(posedge clk);
        $finish;
    end

    initial begin
        #200_000_000;
        $display("ERROR: timeout");
        $finish;
    end

endmodule

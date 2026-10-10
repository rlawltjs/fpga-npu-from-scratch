// ---------------------------------------------------------------------------
// lenet_top.v
//
// LeNet-5 NPU, Stage 1 of the LeNet -> MobileNet -> YOLO11n-mini roadmap.
//
// Dataflow (classic LeNet-5, valid conv, 5x5 kernels, 2x2/stride-2 pool):
//
//   img(1,32,32) --C1(conv,1->6)--> (6,28,28) --S2(pool)--> (6,14,14)
//                --C3(conv,6->16)-> (16,10,10) --S4(pool)--> (16,5,5) [=400 flat]
//                --F5(fc,400->120)-> (120) --F6(fc,120->84)-> (84)
//                --OUT(fc,84->10)-> (10) --ARGMAX--> class index
//
// Only ONE instance each of conv2d_engine / maxpool2x2 / fc_engine /
// argmax_unit exists; the top-level FSM reconfigures each engine's runtime
// in_ch/out_ch/size (or in_dim/out_dim) and re-routes its memory ports via
// combinational muxes for every layer that uses it. Two ping-pong feature
// map buffers (fm_a / fm_b) carry activations between layers so no DMA is
// needed mid-inference.
//
// Weight/bias values are loaded into per-layer ROMs at elaboration time via
// $readmemh (see INIT_FILE parameters below / scripts/gen_weights.py).
// ---------------------------------------------------------------------------
`timescale 1ns/1ps

module lenet_top #(
    parameter DATA_W  = 8,
    parameter ACC_W   = 32,
    parameter ADDR_W  = 16,
    parameter SHIFT   = 0,
    parameter WEIGHTS_DIR = "sim"   // directory containing *.hex files
)(
    input            clk,
    input            rst_n,
    input            start,          // level: hold high to run, drop to reset to idle
    output           busy,
    output           class_valid,
    output [7:0]     class_idx
);

    // ---------------- stage sequencer ----------------
    localparam ST_IDLE   = 4'd0,
               ST_CONV1   = 4'd1,
               ST_POOL1   = 4'd2,
               ST_CONV2   = 4'd3,
               ST_POOL2   = 4'd4,
               ST_FC1     = 4'd5,
               ST_FC2     = 4'd6,
               ST_FC3     = 4'd7,
               ST_ARGMAX  = 4'd8,
               ST_DONE    = 4'd9;

    reg [3:0] stage;
    reg       started;

    assign busy        = (stage != ST_IDLE) && (stage != ST_DONE);
    assign class_valid = (stage == ST_DONE);

    // ---------------- shared engines ----------------
    wire conv_done, pool_done, fc_done, argmax_done;
    reg  conv_start, pool_start, fc_start, argmax_start;
    reg  conv_relu, fc_relu;

    reg [5:0] conv_in_ch, conv_out_ch;
    reg [6:0] conv_in_size;
    reg [5:0] pool_channels;
    reg [6:0] pool_in_size;
    reg [9:0] fc_in_dim;
    reg [8:0] fc_out_dim;

    always @(*) begin
        case (stage)
            ST_CONV1: begin conv_in_ch=6'd1;  conv_out_ch=6'd6;  conv_in_size=7'd32; end
            ST_CONV2: begin conv_in_ch=6'd6;  conv_out_ch=6'd16; conv_in_size=7'd14; end
            default:  begin conv_in_ch=6'd0;  conv_out_ch=6'd0;  conv_in_size=7'd0;  end
        endcase
        conv_relu = 1'b1; // both conv layers use ReLU

        case (stage)
            ST_POOL1: begin pool_channels=6'd6;  pool_in_size=7'd28; end
            ST_POOL2: begin pool_channels=6'd16; pool_in_size=7'd10; end
            default:  begin pool_channels=6'd0;  pool_in_size=7'd0;  end
        endcase

        case (stage)
            ST_FC1: begin fc_in_dim=10'd400; fc_out_dim=9'd120; fc_relu=1'b1; end
            ST_FC2: begin fc_in_dim=10'd120; fc_out_dim=9'd84;  fc_relu=1'b1; end
            ST_FC3: begin fc_in_dim=10'd84;  fc_out_dim=9'd10;  fc_relu=1'b0; end // raw scores
            default: begin fc_in_dim=10'd0;  fc_out_dim=9'd0;   fc_relu=1'b0; end
        endcase
    end

    // start pulses: one clock wide, generated the cycle stage is entered
    always @(*) begin
        conv_start   = (stage == ST_CONV1 || stage == ST_CONV2) && !started;
        pool_start   = (stage == ST_POOL1 || stage == ST_POOL2) && !started;
        fc_start     = (stage == ST_FC1 || stage == ST_FC2 || stage == ST_FC3) && !started;
        argmax_start = (stage == ST_ARGMAX) && !started;
    end

    // ---------------- stage sequencer FSM ----------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage <= ST_IDLE;
            started <= 1'b0;
        end else begin
            case (stage)
                ST_IDLE:  if (start) begin stage <= ST_CONV1; started <= 1'b0; end
                ST_CONV1: if (!started) started <= 1'b1;
                          else if (conv_done)  begin stage <= ST_POOL1; started <= 1'b0; end
                ST_POOL1: if (!started) started <= 1'b1;
                          else if (pool_done)  begin stage <= ST_CONV2; started <= 1'b0; end
                ST_CONV2: if (!started) started <= 1'b1;
                          else if (conv_done)  begin stage <= ST_POOL2; started <= 1'b0; end
                ST_POOL2: if (!started) started <= 1'b1;
                          else if (pool_done)  begin stage <= ST_FC1;   started <= 1'b0; end
                ST_FC1:   if (!started) started <= 1'b1;
                          else if (fc_done)    begin stage <= ST_FC2;   started <= 1'b0; end
                ST_FC2:   if (!started) started <= 1'b1;
                          else if (fc_done)    begin stage <= ST_FC3;   started <= 1'b0; end
                ST_FC3:   if (!started) started <= 1'b1;
                          else if (fc_done)    begin stage <= ST_ARGMAX; started <= 1'b0; end
                ST_ARGMAX: if (!started) started <= 1'b1;
                          else if (argmax_done) stage <= ST_DONE;
                ST_DONE:  if (!start) stage <= ST_IDLE;
                default:  stage <= ST_IDLE;
            endcase
        end
    end

    // ---------------- engine <-> memory bus wires ----------------
    wire [ADDR_W-1:0] conv_ifmap_addr, conv_weight_addr, conv_bias_addr, conv_ofmap_addr;
    wire signed [DATA_W-1:0] conv_ifmap_dout, conv_weight_dout, conv_ofmap_din;
    wire signed [ACC_W-1:0]  conv_bias_dout;
    wire conv_ofmap_we;

    wire [ADDR_W-1:0] pool_ifmap_addr, pool_ofmap_addr;
    wire signed [DATA_W-1:0] pool_ifmap_dout, pool_ofmap_din;
    wire pool_ofmap_we;

    wire [ADDR_W-1:0] fc_ivec_addr, fc_weight_addr, fc_bias_addr, fc_ovec_addr;
    wire signed [DATA_W-1:0] fc_ivec_dout, fc_weight_dout, fc_ovec_din;
    wire signed [ACC_W-1:0]  fc_bias_dout;
    wire fc_ovec_we;

    wire [ADDR_W-1:0] argmax_ivec_addr;
    wire signed [DATA_W-1:0] argmax_ivec_dout;

    // ---------------- physical memories ----------------
    wire signed [DATA_W-1:0] img_dout, fma_dout, fmb_dout;
    wire [ADDR_W-1:0] img_raddr, fma_raddr, fma_waddr, fmb_raddr, fmb_waddr;
    wire signed [DATA_W-1:0] fma_din, fmb_din;
    wire fma_we, fmb_we;

    bram_dp #(.DATA_WIDTH(DATA_W), .DEPTH(1024), .ADDR_WIDTH(ADDR_W),
              .INIT_FILE({WEIGHTS_DIR, "/image.hex"}))
        u_img_mem (.clk(clk), .we(1'b0), .waddr({ADDR_W{1'b0}}), .din({DATA_W{1'b0}}),
                   .raddr(img_raddr), .dout(img_dout));

    bram_dp #(.DATA_WIDTH(DATA_W), .DEPTH(4704), .ADDR_WIDTH(ADDR_W))
        u_fm_a (.clk(clk), .we(fma_we), .waddr(fma_waddr), .din(fma_din),
                .raddr(fma_raddr), .dout(fma_dout));

    bram_dp #(.DATA_WIDTH(DATA_W), .DEPTH(4704), .ADDR_WIDTH(ADDR_W))
        u_fm_b (.clk(clk), .we(fmb_we), .waddr(fmb_waddr), .din(fmb_din),
                .raddr(fmb_raddr), .dout(fmb_dout));

    // conv weight/bias ROMs
    wire signed [DATA_W-1:0] w_conv1_dout, w_conv2_dout;
    wire signed [ACC_W-1:0]  b_conv1_dout, b_conv2_dout;

    bram_dp #(.DATA_WIDTH(DATA_W), .DEPTH(150), .ADDR_WIDTH(ADDR_W),
              .INIT_FILE({WEIGHTS_DIR, "/w_conv1.hex"}))
        u_w_conv1 (.clk(clk), .we(1'b0), .waddr({ADDR_W{1'b0}}), .din({DATA_W{1'b0}}),
                   .raddr(conv_weight_addr), .dout(w_conv1_dout));
    bram_dp #(.DATA_WIDTH(ACC_W), .DEPTH(6), .ADDR_WIDTH(ADDR_W),
              .INIT_FILE({WEIGHTS_DIR, "/b_conv1.hex"}))
        u_b_conv1 (.clk(clk), .we(1'b0), .waddr({ADDR_W{1'b0}}), .din({ACC_W{1'b0}}),
                   .raddr(conv_bias_addr), .dout(b_conv1_dout));

    bram_dp #(.DATA_WIDTH(DATA_W), .DEPTH(2400), .ADDR_WIDTH(ADDR_W),
              .INIT_FILE({WEIGHTS_DIR, "/w_conv2.hex"}))
        u_w_conv2 (.clk(clk), .we(1'b0), .waddr({ADDR_W{1'b0}}), .din({DATA_W{1'b0}}),
                   .raddr(conv_weight_addr), .dout(w_conv2_dout));
    bram_dp #(.DATA_WIDTH(ACC_W), .DEPTH(16), .ADDR_WIDTH(ADDR_W),
              .INIT_FILE({WEIGHTS_DIR, "/b_conv2.hex"}))
        u_b_conv2 (.clk(clk), .we(1'b0), .waddr({ADDR_W{1'b0}}), .din({ACC_W{1'b0}}),
                   .raddr(conv_bias_addr), .dout(b_conv2_dout));

    assign conv_weight_dout = (stage == ST_CONV1) ? w_conv1_dout : w_conv2_dout;
    assign conv_bias_dout   = (stage == ST_CONV1) ? b_conv1_dout : b_conv2_dout;

    // fc weight/bias ROMs
    wire signed [DATA_W-1:0] w_fc1_dout, w_fc2_dout, w_fc3_dout;
    wire signed [ACC_W-1:0]  b_fc1_dout, b_fc2_dout, b_fc3_dout;

    bram_dp #(.DATA_WIDTH(DATA_W), .DEPTH(48000), .ADDR_WIDTH(ADDR_W),
              .INIT_FILE({WEIGHTS_DIR, "/w_fc1.hex"}))
        u_w_fc1 (.clk(clk), .we(1'b0), .waddr({ADDR_W{1'b0}}), .din({DATA_W{1'b0}}),
                 .raddr(fc_weight_addr), .dout(w_fc1_dout));
    bram_dp #(.DATA_WIDTH(ACC_W), .DEPTH(120), .ADDR_WIDTH(ADDR_W),
              .INIT_FILE({WEIGHTS_DIR, "/b_fc1.hex"}))
        u_b_fc1 (.clk(clk), .we(1'b0), .waddr({ADDR_W{1'b0}}), .din({ACC_W{1'b0}}),
                 .raddr(fc_bias_addr), .dout(b_fc1_dout));

    bram_dp #(.DATA_WIDTH(DATA_W), .DEPTH(10080), .ADDR_WIDTH(ADDR_W),
              .INIT_FILE({WEIGHTS_DIR, "/w_fc2.hex"}))
        u_w_fc2 (.clk(clk), .we(1'b0), .waddr({ADDR_W{1'b0}}), .din({DATA_W{1'b0}}),
                 .raddr(fc_weight_addr), .dout(w_fc2_dout));
    bram_dp #(.DATA_WIDTH(ACC_W), .DEPTH(84), .ADDR_WIDTH(ADDR_W),
              .INIT_FILE({WEIGHTS_DIR, "/b_fc2.hex"}))
        u_b_fc2 (.clk(clk), .we(1'b0), .waddr({ADDR_W{1'b0}}), .din({ACC_W{1'b0}}),
                 .raddr(fc_bias_addr), .dout(b_fc2_dout));

    bram_dp #(.DATA_WIDTH(DATA_W), .DEPTH(840), .ADDR_WIDTH(ADDR_W),
              .INIT_FILE({WEIGHTS_DIR, "/w_fc3.hex"}))
        u_w_fc3 (.clk(clk), .we(1'b0), .waddr({ADDR_W{1'b0}}), .din({DATA_W{1'b0}}),
                 .raddr(fc_weight_addr), .dout(w_fc3_dout));
    bram_dp #(.DATA_WIDTH(ACC_W), .DEPTH(10), .ADDR_WIDTH(ADDR_W),
              .INIT_FILE({WEIGHTS_DIR, "/b_fc3.hex"}))
        u_b_fc3 (.clk(clk), .we(1'b0), .waddr({ADDR_W{1'b0}}), .din({ACC_W{1'b0}}),
                 .raddr(fc_bias_addr), .dout(b_fc3_dout));

    assign fc_weight_dout = (stage == ST_FC1) ? w_fc1_dout :
                             (stage == ST_FC2) ? w_fc2_dout : w_fc3_dout;
    assign fc_bias_dout   = (stage == ST_FC1) ? b_fc1_dout :
                             (stage == ST_FC2) ? b_fc2_dout : b_fc3_dout;

    // ---------------- memory port muxing ----------------
    assign img_raddr = conv_ifmap_addr;

    assign fma_raddr = (stage == ST_POOL1 || stage == ST_POOL2) ? pool_ifmap_addr :
                        (stage == ST_FC2)                       ? fc_ivec_addr    :
                        (stage == ST_ARGMAX)                    ? argmax_ivec_addr:
                                                                   {ADDR_W{1'b0}};
    assign fma_waddr  = (stage == ST_CONV1 || stage == ST_CONV2) ? conv_ofmap_addr :
                        (stage == ST_FC1   || stage == ST_FC3)   ? fc_ovec_addr    :
                                                                    {ADDR_W{1'b0}};
    assign fma_din    = (stage == ST_CONV1 || stage == ST_CONV2) ? conv_ofmap_din  : fc_ovec_din;
    assign fma_we     = (stage == ST_CONV1 || stage == ST_CONV2) ? conv_ofmap_we   :
                        (stage == ST_FC1   || stage == ST_FC3)   ? fc_ovec_we      : 1'b0;

    assign fmb_raddr  = (stage == ST_CONV2)                     ? conv_ifmap_addr :
                        (stage == ST_FC1 || stage == ST_FC3)    ? fc_ivec_addr    :
                                                                   {ADDR_W{1'b0}};
    assign fmb_waddr  = (stage == ST_POOL1 || stage == ST_POOL2) ? pool_ofmap_addr : fc_ovec_addr;
    assign fmb_din    = (stage == ST_POOL1 || stage == ST_POOL2) ? pool_ofmap_din  : fc_ovec_din;
    assign fmb_we     = (stage == ST_POOL1 || stage == ST_POOL2) ? pool_ofmap_we   :
                        (stage == ST_FC2)                        ? fc_ovec_we      : 1'b0;

    assign conv_ifmap_dout   = (stage == ST_CONV1) ? img_dout : fmb_dout;
    assign pool_ifmap_dout   = fma_dout;                 // both POOL1 and POOL2 read fm_a
    assign fc_ivec_dout      = (stage == ST_FC2) ? fma_dout : fmb_dout;
    assign argmax_ivec_dout  = fma_dout;                 // final scores always land in fm_a

    // ---------------- engine instances ----------------
    conv2d_engine #(.DATA_W(DATA_W), .ACC_W(ACC_W), .K(5), .ADDR_W(ADDR_W), .SHIFT(SHIFT))
        u_conv (
            .clk(clk), .rst_n(rst_n), .start(conv_start), .done(conv_done),
            .apply_relu(conv_relu),
            .in_ch(conv_in_ch), .out_ch(conv_out_ch), .in_size(conv_in_size),
            .ifmap_addr(conv_ifmap_addr), .ifmap_dout(conv_ifmap_dout),
            .weight_addr(conv_weight_addr), .weight_dout(conv_weight_dout),
            .bias_addr(conv_bias_addr), .bias_dout(conv_bias_dout),
            .ofmap_addr(conv_ofmap_addr), .ofmap_din(conv_ofmap_din), .ofmap_we(conv_ofmap_we)
        );

    maxpool2x2 #(.DATA_W(DATA_W), .ADDR_W(ADDR_W))
        u_pool (
            .clk(clk), .rst_n(rst_n), .start(pool_start), .done(pool_done),
            .channels(pool_channels), .in_size(pool_in_size),
            .ifmap_addr(pool_ifmap_addr), .ifmap_dout(pool_ifmap_dout),
            .ofmap_addr(pool_ofmap_addr), .ofmap_din(pool_ofmap_din), .ofmap_we(pool_ofmap_we)
        );

    fc_engine #(.DATA_W(DATA_W), .ACC_W(ACC_W), .ADDR_W(ADDR_W), .SHIFT(SHIFT))
        u_fc (
            .clk(clk), .rst_n(rst_n), .start(fc_start), .done(fc_done),
            .apply_relu(fc_relu),
            .in_dim(fc_in_dim), .out_dim(fc_out_dim),
            .ivec_addr(fc_ivec_addr), .ivec_dout(fc_ivec_dout),
            .weight_addr(fc_weight_addr), .weight_dout(fc_weight_dout),
            .bias_addr(fc_bias_addr), .bias_dout(fc_bias_dout),
            .ovec_addr(fc_ovec_addr), .ovec_din(fc_ovec_din), .ovec_we(fc_ovec_we)
        );

    wire signed [DATA_W-1:0] argmax_max_val; // unused externally, kept for debug visibility
    argmax_unit #(.DATA_W(DATA_W), .ADDR_W(ADDR_W))
        u_argmax (
            .clk(clk), .rst_n(rst_n), .start(argmax_start), .done(argmax_done),
            .n(8'd10),
            .ivec_addr(argmax_ivec_addr), .ivec_dout(argmax_ivec_dout),
            .max_idx(class_idx), .max_val(argmax_max_val)
        );

endmodule

// ---------------------------------------------------------------------------
// requant_func.vh
//
// Shared fixed-point requantization + optional ReLU function.
// `include this file INSIDE a module body, after the module has declared
// its own `DATA_W`, `ACC_W`, and `SHIFT` parameters.
//
//   acc_in      : 32-bit (ACC_W) signed accumulator value (post bias-add)
//   apply_relu  : 1 = clamp negative values to 0 (ReLU), 0 = pass through
//
// Behavior: arithmetic right-shift by SHIFT bits to rescale back down to
// DATA_W, then saturate to the signed DATA_W range. This mirrors a typical
// post-training-quantization (PTQ) requantization step.
// ---------------------------------------------------------------------------
function signed [DATA_W-1:0] requant_relu;
    input signed [ACC_W-1:0] acc_in;
    input                    apply_relu;
    reg   signed [ACC_W-1:0] shifted;
    begin
        shifted = acc_in >>> SHIFT;
        if (apply_relu && shifted < 0)
            shifted = 0;

        if (shifted > ((1 <<< (DATA_W-1)) - 1))
            requant_relu = (1 <<< (DATA_W-1)) - 1;        // positive saturation
        else if (shifted < -(1 <<< (DATA_W-1)))
            requant_relu = -(1 <<< (DATA_W-1));            // negative saturation
        else
            requant_relu = shifted[DATA_W-1:0];
    end
endfunction

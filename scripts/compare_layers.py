#!/usr/bin/env python3
"""
compare_layers.py

Formal self-checking verification step: runs the golden model, reads the
per-layer RTL dumps produced by tb/tb_dump_layers.v (sim/dbg_*_rtl.hex),
and asserts bit-exact equality at every layer boundary. Exits non-zero
(and prints the first mismatching index/value) if anything disagrees, so
this can be dropped into a CI script.

Prerequisites (run these first):
    python3 scripts/gen_weights.py
    iverilog -g2005 -I rtl -o sim/dump_tb rtl/bram_dp.v rtl/conv2d_engine.v \
        rtl/maxpool2x2.v rtl/fc_engine.v rtl/argmax_unit.v rtl/lenet_top.v \
        tb/tb_dump_layers.v
    vvp sim/dump_tb
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from golden_model import read_hex_i8, read_hex_i32, conv2d_valid, maxpool2x2, fc, SIM_DIR


def read_rtl_dump(filename, n):
    vals = []
    with open(os.path.join(SIM_DIR, filename)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            v = int(line, 16)
            if v >= 0x80:
                v -= 0x100
            vals.append(v)
    return np.array(vals[:n], dtype=np.int64)


def compare(name, golden, rtl_file):
    g = golden.flatten()
    r = read_rtl_dump(rtl_file, g.size)
    ok = np.array_equal(g, r)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:6s} shape={str(golden.shape):16s}", end="")
    if not ok:
        diff = np.where(g != r)[0]
        i0 = diff[0]
        print(f"  first mismatch @ idx={i0}: golden={g[i0]} rtl={r[i0]}  "
              f"({len(diff)}/{g.size} elements differ)")
    else:
        print()
    return ok


def main():
    shift = 4  # must match SHIFT used in the RTL testbench
    image   = read_hex_i8("image.hex",   (1, 32, 32))
    w_conv1 = read_hex_i8("w_conv1.hex", (6, 1, 5, 5))
    b_conv1 = read_hex_i32("b_conv1.hex", (6,))
    w_conv2 = read_hex_i8("w_conv2.hex", (16, 6, 5, 5))
    b_conv2 = read_hex_i32("b_conv2.hex", (16,))
    w_fc1   = read_hex_i8("w_fc1.hex",   (120, 400))
    b_fc1   = read_hex_i32("b_fc1.hex",   (120,))
    w_fc2   = read_hex_i8("w_fc2.hex",   (84, 120))
    b_fc2   = read_hex_i32("b_fc2.hex",   (84,))
    w_fc3   = read_hex_i8("w_fc3.hex",   (10, 84))
    b_fc3   = read_hex_i32("b_fc3.hex",   (10,))

    c1 = conv2d_valid(image, w_conv1, b_conv1, shift, True)
    s2 = maxpool2x2(c1)
    c3 = conv2d_valid(s2, w_conv2, b_conv2, shift, True)
    s4 = maxpool2x2(c3)
    flat = s4.reshape(-1)
    f5 = fc(flat, w_fc1, b_fc1, shift, True)
    f6 = fc(f5, w_fc2, b_fc2, shift, True)
    out = fc(f6, w_fc3, b_fc3, shift, False)

    results = [
        compare("C1",  c1,  "dbg_c1_rtl.hex"),
        compare("S2",  s2,  "dbg_s2_rtl.hex"),
        compare("C3",  c3,  "dbg_c3_rtl.hex"),
        compare("S4",  s4,  "dbg_s4_rtl.hex"),
        compare("F5",  f5,  "dbg_f5_rtl.hex"),
        compare("F6",  f6,  "dbg_f6_rtl.hex"),
        compare("OUT", out, "dbg_out_rtl.hex"),
    ]

    golden_class = int(np.argmax(out))
    print(f"\nGolden model predicted class: {golden_class}")
    print("(compare against the RTL testbench's own printed 'Predicted class index')")

    if all(results):
        print("\nALL LAYERS MATCH -- RTL is numerically bit-exact with the golden model.")
        sys.exit(0)
    else:
        print("\nMISMATCH DETECTED -- see first failing layer above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gen_weights.py

Generates deterministic pseudo-random INT8 weight/bias hex files (and a
INT8 test "image") matching the LeNet-5 NPU's memory layout, for
functional (structural) simulation of lenet_top.v with Icarus Verilog.

These are PLACEHOLDER values for verifying the datapath/control FSM, NOT
weights exported from the trained PyTorch model described in the project
README. To test real classification accuracy:
  1. Train/load the PyTorch LeNet-5 model.
  2. Per-tensor (or per-channel) quantize each weight/bias/activation to
     the same INT8 / INT32-bias fixed-point convention used here.
  3. Write out hex files with the same names/shapes using write_hex() below.

Layouts (must match rtl/lenet_top.v and rtl/conv2d_engine.v / fc_engine.v):
  image.hex   : 1*32*32   = 1024   int8   (ch,row,col)
  w_conv1.hex : 6*1*5*5   = 150    int8   (oc,ic,ky,kx)
  b_conv1.hex : 6                  int32
  w_conv2.hex : 16*6*5*5  = 2400   int8   (oc,ic,ky,kx)
  b_conv2.hex : 16                 int32
  w_fc1.hex   : 120*400   = 48000  int8   (o,i)
  b_fc1.hex   : 120                int32
  w_fc2.hex   : 84*120    = 10080  int8   (o,i)
  b_fc2.hex   : 84                 int32
  w_fc3.hex   : 10*84     = 840    int8   (o,i)
  b_fc3.hex   : 10                 int32
"""
import os
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "sim")
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(seed=42)


def write_hex_i8(filename, values):
    """Write signed 8-bit values as one 2-digit hex (two's complement) per line."""
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w") as f:
        for v in values.flatten():
            v = int(v) & 0xFF
            f.write(f"{v:02x}\n")
    print(f"  {filename:16s} -> {values.size:6d} entries")


def write_hex_i32(filename, values):
    """Write signed 32-bit values as one 8-digit hex (two's complement) per line."""
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w") as f:
        for v in values.flatten():
            v = int(v) & 0xFFFFFFFF
            f.write(f"{v:08x}\n")
    print(f"  {filename:16s} -> {values.size:6d} entries")


def rand_i8(shape, lo=-4, hi=4):
    return rng.integers(lo, hi + 1, size=shape, dtype=np.int64)


def rand_i32(shape, lo=-16, hi=16):
    return rng.integers(lo, hi + 1, size=shape, dtype=np.int64)


print("Generating placeholder LeNet-5 NPU weight/bias/image hex files...")

# Input image: 1x32x32, small positive-ish pixel-like values
image = rng.integers(0, 80, size=(1, 32, 32), dtype=np.int64)
write_hex_i8("image.hex", image)

# C1: 6 out ch, 1 in ch, 5x5
w_conv1 = rand_i8((6, 1, 5, 5))
b_conv1 = rand_i32((6,))
write_hex_i8("w_conv1.hex", w_conv1)
write_hex_i32("b_conv1.hex", b_conv1)

# C3: 16 out ch, 6 in ch, 5x5
w_conv2 = rand_i8((16, 6, 5, 5))
b_conv2 = rand_i32((16,))
write_hex_i8("w_conv2.hex", w_conv2)
write_hex_i32("b_conv2.hex", b_conv2)

# F5: 120x400
w_fc1 = rand_i8((120, 400))
b_fc1 = rand_i32((120,))
write_hex_i8("w_fc1.hex", w_fc1)
write_hex_i32("b_fc1.hex", b_fc1)

# F6: 84x120
w_fc2 = rand_i8((84, 120))
b_fc2 = rand_i32((84,))
write_hex_i8("w_fc2.hex", w_fc2)
write_hex_i32("b_fc2.hex", b_fc2)

# OUT: 10x84
w_fc3 = rand_i8((10, 84))
b_fc3 = rand_i32((10,))
write_hex_i8("w_fc3.hex", w_fc3)
write_hex_i32("b_fc3.hex", b_fc3)

print(f"Done. Files written to: {os.path.abspath(OUT_DIR)}")

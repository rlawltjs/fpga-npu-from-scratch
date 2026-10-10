#!/usr/bin/env python3
"""
golden_model.py

Bit-exact (integer, fixed-point) NumPy re-implementation of what
lenet_top.v actually computes: same conv/pool/fc order of operations,
same requantization rule (acc >>> SHIFT, saturate to int8, optional ReLU).

Purpose: verification, NOT a reference "float" LeNet-5. This script reads
the exact same *.hex files the RTL simulation reads (sim/*.hex) and
reproduces the RTL's integer arithmetic in Python so the two can be
compared layer-by-layer. If RTL and golden model disagree, the bug is in
the RTL (address mapping, off-by-one loop bound, sign-extension, etc.)
since the golden model is the simpler, trusted implementation.

Usage:
    python3 scripts/golden_model.py --shift 4
(must match the SHIFT parameter used when the RTL testbench was run)
"""
import argparse
import os
import numpy as np

SIM_DIR = os.path.join(os.path.dirname(__file__), "..", "sim")


def read_hex_i8(filename, shape):
    path = os.path.join(SIM_DIR, filename)
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16)
            if v >= 0x80:       # sign-extend 8-bit two's complement
                v -= 0x100
            vals.append(v)
    arr = np.array(vals, dtype=np.int64).reshape(shape)
    return arr


def read_hex_i32(filename, shape):
    path = os.path.join(SIM_DIR, filename)
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16)
            if v >= 0x80000000:  # sign-extend 32-bit two's complement
                v -= 0x100000000
            vals.append(v)
    arr = np.array(vals, dtype=np.int64).reshape(shape)
    return arr


def requant_relu(acc, shift, apply_relu):
    """Exactly mirrors requant_func.vh: acc >>> SHIFT, optional ReLU, saturate to int8."""
    shifted = acc >> shift   # numpy int64 '>>' on negative values is arithmetic (floor), same as Verilog >>>
    if apply_relu:
        shifted = np.maximum(shifted, 0)
    return np.clip(shifted, -128, 127).astype(np.int64)


def conv2d_valid(ifmap, weight, bias, shift, apply_relu):
    """ifmap: (in_ch, H, W), weight: (out_ch, in_ch, K, K), bias: (out_ch,)"""
    in_ch, H, W = ifmap.shape
    out_ch, _, K, _ = weight.shape
    out_size = H - K + 1
    ofmap = np.zeros((out_ch, out_size, out_size), dtype=np.int64)
    for oc in range(out_ch):
        for oy in range(out_size):
            for ox in range(out_size):
                patch = ifmap[:, oy:oy+K, ox:ox+K]           # (in_ch,K,K)
                acc = int(np.sum(patch.astype(np.int64) * weight[oc].astype(np.int64)))
                acc += int(bias[oc])
                ofmap[oc, oy, ox] = requant_relu(acc, shift, apply_relu)
    return ofmap


def maxpool2x2(ifmap):
    ch, H, W = ifmap.shape
    out_size = H // 2
    ofmap = np.zeros((ch, out_size, out_size), dtype=np.int64)
    for c in range(ch):
        for oy in range(out_size):
            for ox in range(out_size):
                window = ifmap[c, oy*2:oy*2+2, ox*2:ox*2+2]
                ofmap[c, oy, ox] = np.max(window)
    return ofmap


def fc(ivec, weight, bias, shift, apply_relu):
    """ivec: (in_dim,), weight: (out_dim,in_dim), bias:(out_dim,)"""
    out_dim, in_dim = weight.shape
    ovec = np.zeros((out_dim,), dtype=np.int64)
    for o in range(out_dim):
        acc = int(np.sum(ivec.astype(np.int64) * weight[o].astype(np.int64))) + int(bias[o])
        ovec[o] = requant_relu(acc, shift, apply_relu)
    return ovec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=int, default=4, help="must match RTL SHIFT parameter")
    args = ap.parse_args()
    shift = args.shift

    print(f"Loading hex files from {os.path.abspath(SIM_DIR)} (SHIFT={shift}) ...")
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

    c1 = conv2d_valid(image, w_conv1, b_conv1, shift, apply_relu=True)
    print(f"  C1  out shape {c1.shape}  range [{c1.min()},{c1.max()}]")
    s2 = maxpool2x2(c1)
    print(f"  S2  out shape {s2.shape}  range [{s2.min()},{s2.max()}]")
    c3 = conv2d_valid(s2, w_conv2, b_conv2, shift, apply_relu=True)
    print(f"  C3  out shape {c3.shape}  range [{c3.min()},{c3.max()}]")
    s4 = maxpool2x2(c3)
    print(f"  S4  out shape {s4.shape}  range [{s4.min()},{s4.max()}]")

    flat = s4.reshape(-1)   # channel-major, matches RTL addr = ch*(sz*sz)+row*sz+col
    f5 = fc(flat, w_fc1, b_fc1, shift, apply_relu=True)
    print(f"  F5  out shape {f5.shape}  range [{f5.min()},{f5.max()}]")
    f6 = fc(f5, w_fc2, b_fc2, shift, apply_relu=True)
    print(f"  F6  out shape {f6.shape}  range [{f6.min()},{f6.max()}]")
    out = fc(f6, w_fc3, b_fc3, shift, apply_relu=False)
    print(f"  OUT scores: {out.tolist()}")

    pred = int(np.argmax(out))
    print("=====================================================")
    print(f" Golden model predicted class : {pred}")
    print("=====================================================")


if __name__ == "__main__":
    main()

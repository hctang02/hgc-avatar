import os
import math
import argparse
import json
import torch
import numpy as np
from bitstream.range_coder import RangeCoder
import time  # 导入time模块

# ========== 全局参数 ==========
MIN_SCALE_NN_WEIGHTS_BIAS = 1e-3
POSSIBLE_SCALE_NN = torch.logspace(
    math.log10(MIN_SCALE_NN_WEIGHTS_BIAS), 3, steps=2 ** 16 - 1, device='cpu'
)
POSSIBLE_Q_STEP_NN = 10. ** torch.linspace(-5, 0, 11, device='cpu')
Q_step_list = np.logspace(start=-7, stop=0, num=10, endpoint=True, base=10.0)

# ========== 工具函数 ==========
def get_ac_max_val_nn(state, q_index):
    q_step = Q_step_list[q_index]
    param = [torch.round(v.float() / q_step).flatten() for k, v in state.items()]
    param_quant = torch.cat(param).flatten()
    return int(torch.ceil(param_quant.abs().max() + 2).item())


def write_header(header_path, q_index, scale_index, ac_max_val):
    byte_to_write = b''
    byte_to_write += (10).to_bytes(2, 'big')                      # header size
    byte_to_write += q_index.to_bytes(2, 'big')
    byte_to_write += scale_index.to_bytes(2, 'big')
    byte_to_write += ac_max_val.to_bytes(4, 'big')

    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, 'wb') as fout:
        fout.write(byte_to_write)

    assert os.path.getsize(header_path) == 10, "Header size mismatch!"


def read_header(header_path):
    with open(header_path, 'rb') as f:
        bitstream = f.read()
    ptr = 0
    n_bytes_header = int.from_bytes(bitstream[ptr: ptr + 2], 'big')
    q_index = int.from_bytes(bitstream[2:4], 'big')
    scale_index = int.from_bytes(bitstream[4:6], 'big')
    ac_max_val = int.from_bytes(bitstream[6:10], 'big')
    return n_bytes_header, q_index, scale_index, ac_max_val


# ========== 编解码函数 ==========
def encode(state, header_path, bits_path, q_index):
    encoding_start_time = time.time()

    q_step = Q_step_list[q_index]
    ac_max_val = get_ac_max_val_nn(state, q_index)
    print("ac_max_val is", ac_max_val)
    range_coder = RangeCoder(0, ac_max_val)

    q_weights = []
    for k, v in state.items():
        q_weights.append(torch.round(v.float() / q_step).flatten())
    q_weights = torch.cat(q_weights).flatten()

    floating_point_scale_weight = q_weights.std().item() / math.sqrt(2)
    scale_index_weight = int(torch.argmin((POSSIBLE_SCALE_NN - floating_point_scale_weight).abs()).item())
    scale_weight = POSSIBLE_SCALE_NN[scale_index_weight]

    range_coder.encode(
        bits_path,
        q_weights.cpu(),
        torch.zeros_like(q_weights),
        scale_weight * torch.ones_like(q_weights),
        CHW=None
    )

    write_header(header_path, q_index, scale_index_weight, ac_max_val)

    encoding_end_time = time.time()
    encoding_duration = encoding_end_time - encoding_start_time
    print(f"编码时长: {encoding_duration:.2f} 秒")

    return os.path.getsize(bits_path), encoding_duration


def decode(empty_state, header_path, bits_path):
    decoding_start_time = time.time()

    _, q_index, scale_index, ac_max_val = read_header(header_path)
    q_step = Q_step_list[q_index]
    scale_weight = POSSIBLE_SCALE_NN[scale_index]
    range_decoder = RangeCoder(0, ac_max_val)
    range_decoder.load_bitstream(bits_path)

    decode_state = {}
    for k, v in empty_state.items():
        decoded_param = range_decoder.decode(
            torch.zeros_like(v.flatten()),
            torch.ones_like(v.flatten()) * scale_weight
        )
        decode_state[k] = decoded_param.reshape_as(v) * q_step

    decoding_end_time = time.time()
    decoding_duration = decoding_end_time - decoding_start_time
    print(f"解码时长: {decoding_duration:.2f} 秒")

    return decode_state, decoding_duration


# ========== 两个误差指标（MSE + Max Error） ==========
def compute_diff_metrics(original_state, decoded_state):
    mse = 0.0
    max_err = 0.0
    total = 0

    for k in original_state.keys():
        orig = original_state[k].float()
        dec = decoded_state[k].float()
        diff = orig - dec

        mse += (diff ** 2).sum().item()
        max_err = max(max_err, diff.abs().max().item())
        total += orig.numel()

    mse /= total
    return mse, max_err


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--q_index', type=int, default=5)
    parser.add_argument('--batch_name', type=str, default='batch_800000')
    parser.add_argument('--input_ckpt', type=str)
    parser.add_argument('--output_ckpt', type=str)
    parser.add_argument('--bitstream_dir', type=str)
    parser.add_argument('--metrics_path', type=str)
    args = parser.parse_args()

    direct_mode = args.input_ckpt or args.output_ckpt or args.bitstream_dir
    if direct_mode:
        if not all((args.input_ckpt, args.output_ckpt, args.bitstream_dir)):
            parser.error('--input_ckpt, --output_ckpt and --bitstream_dir are required together')
        ckpt_path = args.input_ckpt
        output_ckpt = args.output_ckpt
        save_dir = os.path.dirname(output_ckpt) or '.'
        bin_dir = args.bitstream_dir
    else:
        if not args.root_dir or not args.data_name:
            parser.error('--root_dir and --data_name are required in legacy layout mode')
        ckpt_path = os.path.join(args.root_dir, 'before_compress', args.data_name, args.batch_name, 'net.pt')
        save_dir = os.path.join(args.root_dir, 'after_compress', args.data_name, args.batch_name)
        output_ckpt = os.path.join(save_dir, 'net.pt')
        bin_dir = os.path.join(args.root_dir, 'bin', args.data_name, 'avatar')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['avatar_net']
    dict1 = {k: v for k, v in state.items() if "num_batches_tracked" in k}
    dict2 = {k: v for k, v in state.items() if "num_batches_tracked" not in k}

    header_path = os.path.join(bin_dir, f"{args.q_index}_header")
    bits_path = os.path.join(bin_dir, f"{args.q_index}_bits")

    # 编码
    _, encoding_time = encode(dict2, header_path, bits_path, args.q_index)

    # 解码
    decode_state, decoding_time = decode(dict2, header_path, bits_path)

    # ====== 误差统计 ======
    mse, max_err = compute_diff_metrics(dict2, decode_state)
    print("\n====== 网络参数压缩误差 ======")
    print(f"MSE:       {mse:.6e}")
    print(f"Max error: {max_err:.6e}")
    print("================================\n")

    # BatchNorm counters are not entropy-coded, but they still belong inside the
    # avatar state dict.  The legacy script merged them at the checkpoint root,
    # producing a checkpoint that could not be loaded strictly.
    decode_state.update(dict1)
    final_result = {
        "epoch_idx": ckpt["epoch_idx"],
        "iter_idx": ckpt["iter_idx"],
        "avatar_net": decode_state
    }
    torch.save(final_result, output_ckpt)
    print("✅ 编码 & 解码完成，保存到:", output_ckpt)

    metrics = {
        "q_index": args.q_index,
        "q_step": float(Q_step_list[args.q_index]),
        "source_checkpoint_bytes": os.path.getsize(ckpt_path),
        "decoded_checkpoint_bytes": os.path.getsize(output_ckpt),
        "header_bytes": os.path.getsize(header_path),
        "bitstream_bytes": os.path.getsize(bits_path),
        "mse": mse,
        "max_absolute_error": max_err,
        "encoding_seconds": encoding_time,
        "decoding_seconds": decoding_time,
    }
    if args.metrics_path:
        metrics_dir = os.path.dirname(args.metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(args.metrics_path, 'w', encoding='utf-8') as fout:
            json.dump(metrics, fout, ensure_ascii=False, indent=2)

    print(f"\n网络参数总编码时长: {encoding_time:.2f} 秒")
    print(f"网络参数总解码时长: {decoding_time:.2f} 秒")


if __name__ == '__main__':
    main()

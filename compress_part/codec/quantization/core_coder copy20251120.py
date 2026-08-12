import os
import math
import argparse
import torch
import numpy as np
from bitstream.range_coder import RangeCoder

# ========== 全局参数 ==========
MIN_SCALE_NN_WEIGHTS_BIAS = 1e-3
POSSIBLE_SCALE_NN = 10 ** torch.linspace(MIN_SCALE_NN_WEIGHTS_BIAS, 1e3, steps=2 ** 16 - 1, device='cpu')
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


def encode(state, header_path, bits_path, q_index):
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
    return os.path.getsize(bits_path)


def decode(empty_state, header_path, bits_path):
    _, q_index, scale_index, ac_max_val = read_header(header_path)
    q_step = Q_step_list[q_index]
    scale_weight = POSSIBLE_SCALE_NN[scale_index]
    range_decoder = RangeCoder(0, ac_max_val)
    range_decoder.load_bitstream(bits_path)

    decode_state = {}
    for k, v in empty_state.items():
        decoded_param = range_decoder.decode(torch.zeros_like(v.flatten()), torch.ones_like(v.flatten()) * scale_weight)
        decode_state[k] = decoded_param.reshape_as(v) * q_step
    return decode_state


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--q_index', type=int, default=5)
    parser.add_argument('--batch_name', type=str, default='batch_800000')
    args = parser.parse_args()

    ckpt_path = os.path.join(args.root_dir, 'before_compress', args.data_name, args.batch_name, 'net.pt')
    save_dir = os.path.join(args.root_dir, 'after_compress', args.data_name, args.batch_name)
    bin_dir = os.path.join(args.root_dir, 'bin', args.data_name, 'avatar')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['avatar_net']
    dict1 = {k: v for k, v in state.items() if "num_batches_tracked" in k}
    dict2 = {k: v for k, v in state.items() if "num_batches_tracked" not in k}

    header_path = os.path.join(bin_dir, f"{args.q_index}_header")
    bits_path = os.path.join(bin_dir, f"{args.q_index}_bits")

    encode(dict2, header_path, bits_path, args.q_index)
    decode_state = decode(dict2, header_path, bits_path)

    final_result = {
        "epoch_idx": ckpt["epoch_idx"],
        "iter_idx": ckpt["iter_idx"],
        "avatar_net": decode_state
    }
    merged_dict = {**dict1, **final_result}
    torch.save(merged_dict, os.path.join(save_dir, 'net.pt'))
    print("✅ 编码 & 解码完成，保存到:", os.path.join(save_dir, 'net.pt'))


if __name__ == '__main__':
    main()

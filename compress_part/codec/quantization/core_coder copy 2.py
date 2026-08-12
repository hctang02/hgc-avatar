import math
import os
import torch
from bitstream.range_coder import RangeCoder
import numpy as np

# Avoid numerical instability when measuring the rate of the NN parameters
MIN_SCALE_NN_WEIGHTS_BIAS = 1e-3
# List of all possible scales when coding a MLP
POSSIBLE_SCALE_NN = 10 ** torch.linspace(MIN_SCALE_NN_WEIGHTS_BIAS, 1e3, steps=2 ** 16 - 1, device='cpu')
# List of all possible quantization steps when coding a MLP
POSSIBLE_Q_STEP_NN = 10. ** torch.linspace(-5, 0, 11, device='cpu')

#Q_step_list = np.logspace(start=-9, stop=0, num=10, endpoint=True, base=10.0)
Q_step_list = np.logspace(start=-7, stop=0, num=10, endpoint=True, base=10.0)

def get_ac_max_val_nn(state, q_index):
    q_step = Q_step_list[q_index]
    param = []

    for k, v in state.items():
        param.append(torch.round(v.float() / q_step).flatten())#将每个值除以 q_step 并进行四舍五入操作，然后将结果展平后添加到 param 列表中。

    param_quant = torch.cat(param).flatten()#将 param 列表中的张量连接起来,并使用 flatten() 函数将连接后的张量展平
  
    AC_MAX_VAL = int(torch.ceil(param_quant.abs().max() + 2).item())#torch.ceil() 函数对最大值进行向上取整，并加上 2

    return AC_MAX_VAL

def write_header(header_path, q_index, scale_index, ac_max_val):
    n_bytes_header = 0
    n_bytes_header += 2  # Number of bytes header
    n_bytes_header += 2  # Number of q_index
    n_bytes_header += 2  # Number of scale_index
    n_bytes_header += 4  # Number of ac_max_val

    byte_to_write = b''
    byte_to_write += n_bytes_header.to_bytes(2, byteorder='big', signed=False)
    byte_to_write += q_index.to_bytes(2, byteorder='big', signed=False)
    byte_to_write += scale_index.to_bytes(2, byteorder='big', signed=False)
    byte_to_write += ac_max_val.to_bytes(4, byteorder='big', signed=False)

    with open(header_path, 'wb') as fout:
        fout.write(byte_to_write)

    assert n_bytes_header == os.path.getsize(header_path),\
        'Invalid number of bytes in header!'

def read_header(header_path):
    with open(header_path, 'rb') as fin:
        bitstream = fin.read()

    ptr = 0
    n_bytes_header = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2

    q_index = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2

    scale_index = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2

    ac_max_val = int.from_bytes(bitstream[ptr: ptr + 4], byteorder='big', signed=False)
    ptr += 4

    return n_bytes_header, q_index, scale_index, ac_max_val

def encode(state, header_path, bits_path, q_index):
    q_step = Q_step_list[q_index]
    # print("333333333333")
    ac_max_val = get_ac_max_val_nn(state, q_index)
    print("ac_max_val is ", ac_max_val)
    range_coder = RangeCoder(0, ac_max_val)
    # print("22222222")
    q_weights = []
    for k, v in state.items():
        q_weights.append(torch.round(v.float() / q_step).flatten())


    q_weights = torch.cat(q_weights).flatten()
    floating_point_scale_weight = q_weights.std().item() / math.sqrt(2)#标准差
    scale_index_weight = int(torch.argmin((POSSIBLE_SCALE_NN - floating_point_scale_weight).abs()).item())
    scale_weight = POSSIBLE_SCALE_NN[scale_index_weight]
    q_weights = q_weights.cpu()

    range_coder.encode(
        bits_path,
        q_weights,
        torch.zeros_like(q_weights),
        scale_weight * torch.ones_like(q_weights),
        CHW=None,  # No wavefront coding for the weights
    )
    n_bytes = os.path.getsize(bits_path)

    write_header(header_path, q_index, scale_index_weight, ac_max_val)

    return n_bytes

def decode(empty_state, header_path, bits_path):
    _, q_index, scale_index, ac_max_val = read_header(header_path)
    q_step = Q_step_list[q_index]
    scale_weight = POSSIBLE_SCALE_NN[scale_index]

    range_decoder = RangeCoder(0, ac_max_val)
    range_decoder.load_bitstream(bits_path)
    decode_state = dict()
    for k, v in empty_state.items():
        cur_param = range_decoder.decode(torch.zeros_like(v.flatten()), torch.ones_like(v.flatten()) * scale_weight)
        decode_state[k] = cur_param.reshape_as(v)  * q_step
    return decode_state

if __name__ == '__main__':
    # 1. net.pt的压缩
    # header_path = "header"
    # bits_path = "bits"
    # ./results/avatarrex_zzr/avatar-compress/batch_800000/net.pt
    name = "exercise"

    # ckpt = torch.load(f"/mnt/data-ssd1/ruoke/pro/lora-Owlii/lora/{name}/1-cnvrtd.ckpt",map_location="cpu")
    ckpt = torch.load("../results/avatarrex_zzr/avatar-compress/batch_800000/net.pt",map_location="cpu")
    #state = ckpt["network"]
    state = ckpt['avatar_net']  # Accessing the 'avatar_net' part

    original_dict = state

    dict1 = {}
    dict2 = {}

    for key, value in original_dict.items():
        if "num_batches_tracked" in key:
            dict1[key] = value
        else:
            dict2[key] = value

    q_index = 5#4  6
    header_path = f"../results/avatarrex_zzr/avatar-compress/batch_800000/q/{q_index}_header"
    # header_path = f"/mnt/data-ssd1/ruoke/pro/lora-Owlii/lora/{name}/q/{q_index}_header"
    bits_path = f"../results/avatarrex_zzr/avatar-compress/batch_800000/q/{q_index}_bits"
    # bits_path = f"/mnt/data-ssd1/ruoke/pro/lora-Owlii/lora/{name}/q/{q_index}_bits"
    ac_max_val = get_ac_max_val_nn(dict2, q_index)
    encode(dict2, header_path, bits_path, q_index)
    decode_state = decode(dict2, header_path, bits_path)
    final_result = {
        "epoch_idx": ckpt["epoch_idx"],
        "iter_idx": ckpt["iter_idx"],
        "avatar_net": decode_state
    }
    merged_dict = {**dict1, **final_result}

    #decode_ckpt = {"network": merged_dict}  # 创建保存的字典，键为 "network"，值为 state
    #print(merged_dict)
    
    decode_ckpt = merged_dict

    # torch.save(decode_ckpt, f"/mnt/data-ssd1/ruoke/pro/lora-Owlii/lora/{name}/q/{q_index}.ckpt")
    torch.save(decode_ckpt, f"../results/avatarrex_zzr/avatar-compress/batch_800000/q/{q_index}.ckpt")



    # # 2. optm.pt的压缩
    # # header_path = "header"
    # # bits_path = "bits"
    # # ./results/avatarrex_zzr/avatar-compress/batch_800000/net.pt
    # # name = "exercise"

    # # ckpt = torch.load(f"/mnt/data-ssd1/ruoke/pro/lora-Owlii/lora/{name}/1-cnvrtd.ckpt",map_location="cpu")
    # ckpt = torch.load("../results/avatarrex_zzr/avatar-compress/batch_800000/optm.pt",map_location="cpu")
    # #state = ckpt["network"]
    # avatarnet = ckpt['avatar_net']  # Accessing the 'avatar_net' part
    # state = avatarnet['state'] 
    # param_groups = avatarnet['param_groups']

    # q_index = 9#4  6

    # state_result = {}
    # for key, value in state.items():
    #     original_dict = state[key]
    #     number = key
    #     dict1 = {}
    #     dict2 = {}

    #     for key, value in original_dict.items():
    #         dict2[key] = value
    #     ac_max_val = get_ac_max_val_nn(dict2, q_index)
    #     header_path = f"../results/avatarrex_zzr/avatar-compress/batch_800000/q2/{q_index}_{number}_header"
    #     bits_path = f"../results/avatarrex_zzr/avatar-compress/batch_800000/q2/{q_index}_{number}_bits"
    #     print(dict2.keys())
    #     encode(dict2, header_path, bits_path, q_index)
    #     # print("1111111111111")
    #     decode_state = decode(dict2, header_path, bits_path)
    #     state_result[number] = decode_state
    
    # # dict3 = {}
    # # original_dict = param_groups
    # # for key, value in original_dict.items():
    # #     if value==None:
    # #         dict3[key] = None
    # #     else:
    # #         dict3[key] = torch.tensor(value)
    # # print(dict3)
    # # ac_max_val = get_ac_max_val_nn(dict3, q_index)
    # # header_path = f"../results/avatarrex_zzr/avatar-compress/batch_800000/q2/{q_index}_param_header"
    # # bits_path = f"../results/avatarrex_zzr/avatar-compress/batch_800000/q2/{q_index}_param_bits"
    # # encode(dict3, header_path, bits_path, q_index)
    # # param_groups_result = decode(dict3, header_path, bits_path)
    

    # final_result= {
    #     "avatar_net": {"state": state_result},
    #     "param_groups": param_groups
    # }
    # merged_dict = {**dict1, **final_result}

    # #decode_ckpt = {"network": merged_dict}  # 创建保存的字典，键为 "network"，值为 state
    # #print(merged_dict)
    
    # decode_ckpt = merged_dict

    # # torch.save(decode_ckpt, f"/mnt/data-ssd1/ruoke/pro/lora-Owlii/lora/{name}/q/{q_index}.ckpt")
    # torch.save(decode_ckpt, f"../results/avatarrex_zzr/avatar-compress/batch_800000/q2/{q_index}.ckpt")

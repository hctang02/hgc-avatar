import os
import numpy as np
import heapq
import argparse

# ========== 霍夫曼编码工具类 ==========
class Node:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequency):
    priority_queue = [Node(symbol, freq) for symbol, freq in frequency.items()]
    heapq.heapify(priority_queue)
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = Node(freq=left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)
    return priority_queue[0]

def generate_huffman_codes(root, code="", codes={}):
    if root.symbol is not None:
        codes[root.symbol] = code
    if root.left is not None:
        generate_huffman_codes(root.left, code + "0", codes)
    if root.right is not None:
        generate_huffman_codes(root.right, code + "1", codes)
    return codes

# ========== 编解码函数 ==========
def encode_and_decode(data, num_bins, file_name_prefix, bin_dir):
    # 步骤 1: 对数据进行归一化
    min_val, max_val = np.min(data), np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)

    # 步骤 2: 将归一化值映射到离散符号
    symbols = np.floor(normalized_data * num_bins).astype(int)
    symbols = np.clip(symbols, 0, num_bins - 1)

    # 步骤 3: 构建符号频率分布
    frequency = {}
    for symbol in symbols:
        frequency[symbol] = frequency.get(symbol, 0) + 1

    # 步骤 4: 构建霍夫曼树与编码表
    class Node:
        def __init__(self, symbol=None, freq=0):
            self.symbol = symbol
            self.freq = freq
            self.left = None
            self.right = None
        def __lt__(self, other):
            return self.freq < other.freq

    def build_huffman_tree(frequency):
        priority_queue = [Node(symbol, freq) for symbol, freq in frequency.items()]
        heapq.heapify(priority_queue)
        while len(priority_queue) > 1:
            left = heapq.heappop(priority_queue)
            right = heapq.heappop(priority_queue)
            merged = Node(freq=left.freq + right.freq)
            merged.left = left
            merged.right = right
            heapq.heappush(priority_queue, merged)
        return priority_queue[0]

    def generate_huffman_codes(root, code="", codes={}):
        if root.symbol is not None:
            codes[root.symbol] = code
        if root.left is not None:
            generate_huffman_codes(root.left, code + "0", codes)
        if root.right is not None:
            generate_huffman_codes(root.right, code + "1", codes)
        return codes

    huffman_tree = build_huffman_tree(frequency)
    huffman_codes = generate_huffman_codes(huffman_tree)

    # 步骤 5: 编码符号成 bit 串
    encoded_data = ''.join(huffman_codes[symbol] for symbol in symbols)
    bit_len = len(encoded_data)

    # 步骤 6: 保存 bit_len + 编码数据（转字节）
    with open(os.path.join(bin_dir, f"{file_name_prefix}.bin"), "wb") as f:
        f.write(bit_len.to_bytes(4, byteorder='big'))  # 保存比特长度
        byte_data = bytearray(
            int(encoded_data[i:i+8].ljust(8, '0'), 2)
            for i in range(0, bit_len, 8)
        )
        f.write(bytes(byte_data))

    # 步骤 7: 解码
    with open(os.path.join(bin_dir, f"{file_name_prefix}.bin"), "rb") as f:
        bit_len = int.from_bytes(f.read(4), byteorder='big')  # 读取长度
        byte_data = f.read()

    binary_data = ''.join(f'{byte:08b}' for byte in byte_data)[:bit_len]

    decoded_symbols = []
    current_code = ""
    reverse_huffman_codes = {v: k for k, v in huffman_codes.items()}

    for bit in binary_data:
        current_code += bit
        if current_code in reverse_huffman_codes:
            decoded_symbols.append(reverse_huffman_codes[current_code])
            current_code = ""

    if len(decoded_symbols) != len(symbols):
        print(f"⚠️ 警告：解码符号数量与原始不一致！")
        decoded_symbols = decoded_symbols[:len(symbols)]

    # 步骤 8: 逆归一化
    decoded_data = []
    for symbol in decoded_symbols:
        midpoint = symbol / num_bins
        original_value = midpoint * (max_val - min_val) + min_val
        decoded_data.append(original_value)

    return np.array(decoded_data)


# ========== 主程序 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="根目录")
    parser.add_argument("--data_name", type=str, required=True, help="数据名称")
    parser.add_argument("--num_bins", type=int, default=100000, help="Huffman编码的bin数")
    args = parser.parse_args()

    root_dir = args.root_dir
    data_name = args.data_name
    num_bins = args.num_bins

    input_file_path = f'{root_dir}/before_compress/{data_name}/smpl_params.npz'
    output_dir = f'{root_dir}/after_compress/{data_name}/'
    bin_dir = f'{root_dir}/bin/{data_name}/smpl_params/'
    txt_dir = f'{root_dir}/test/{data_name}/smpl_params/'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)
    os.makedirs(f'{txt_dir}/encode/', exist_ok=True)
    os.makedirs(f'{txt_dir}/decode/', exist_ok=True)

    with np.load(input_file_path) as data:
        print("存储的数组键名:", data.files)
        names_shapes = {}
        for key in data.files:
            array_data = data[key]
            names_shapes[key] = array_data.shape
            filename = os.path.join(txt_dir, f"encode/{key}.txt")
            np.savetxt(filename, array_data, fmt='%.6f')
        print("\n构建的 names_shapes 字典:")
        print(names_shapes)

    for name, shape in names_shapes.items():
        data = np.loadtxt(os.path.join(txt_dir, f'encode/{name}.txt'), dtype=np.float128)
        print(f"原始 {name} 数据:", data)

        decoded_data = encode_and_decode(data.flatten(), num_bins, name, bin_dir)
        decoded_data = decoded_data.reshape(shape)

        decode_txt_path = os.path.join(txt_dir, f'decode/{name}.txt')
        if name == "betas":
            with open(decode_txt_path, 'w') as f:
                np.savetxt(f, decoded_data, fmt='%0.6f', delimiter=' ', newline=' ')
        elif name in ["global_orient", "transl", "body_pose", "jaw_pose", "expression", "left_hand_pose", "right_hand_pose"]:
            with open(decode_txt_path, 'w') as f:
                for row in decoded_data:
                    f.write(" ".join([f"{val:.6f}" for val in row]) + "\n")
        print(f"解码后的 {name} 数据:", np.round(decoded_data, 6))

    # 保存为 npz
    arrays_to_save = {}
    for name, shape in names_shapes.items():
        decode_txt_path = os.path.join(txt_dir, f'decode/{name}.txt')
        if os.path.exists(decode_txt_path):
            data = np.loadtxt(decode_txt_path, dtype=np.float64)
            restored_data = data.reshape(shape)
            arrays_to_save[name] = restored_data
    np.savez(os.path.join(output_dir, 'smpl_params.npz'), **arrays_to_save)

if __name__ == "__main__":
    main()

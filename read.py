import torch

def load_and_view_first_layer(path, num_print=30):
    print(f"Loading: {path}")
    ckpt = torch.load(path, map_location='cpu')

    if "avatar_net" not in ckpt:
        print("❌ avatar_net 不存在")
        return

    state = ckpt["avatar_net"]

    # 自动找到第一层
    first_key = list(state.keys())[0]
    first_param = state[first_key]

    print("\n===== 第一层参数 =====")
    print(f"层名: {first_key}")
    print(f"shape: {tuple(first_param.shape)}")

    flat = first_param.flatten()
    print(f"\n前 {num_print} 个数值:")
    print(flat[:num_print])

    return first_param


if __name__ == "__main__":
    path = "/mnt/ssd2tB/haocheng/AnimatableGaussians/compress_part/data/lbn1_point1_q=4/after_compress/output-lbn1-test/batch_800000/net.pt"
    load_and_view_first_layer(path)
    path2 = "/mnt/ssd2tB/haocheng/AnimatableGaussians/compress_part/data/lbn1_point2_q=5/after_compress/output-lbn1-test/batch_800000/net-5.pt"
    load_and_view_first_layer(path2)
    path3 = "/mnt/ssd2tB/haocheng/AnimatableGaussians/compress_part/data/lbn1_point3_q=6/after_compress/output-lbn1-test/batch_800000/net-6.pt"
    load_and_view_first_layer(path3)
    path4 = "/mnt/ssd2tB/haocheng/AnimatableGaussians/compress_part/data/lbn1_point4_q=7/after_compress/output-lbn1-test/batch_800000/net-7.pt"
    load_and_view_first_layer(path4)
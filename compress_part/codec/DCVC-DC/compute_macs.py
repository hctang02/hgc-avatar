import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

from src.models.image_model import IntraNoAR
from src.models.video_model import DMC
# model = IntraNoAR(ec_thread=False, stream_part=1, inplace=True)
# print(model)

# # === 包装 I 帧模型 ===
class WrappedIntraNet(IntraNoAR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 你已经有 q_basic_enc 参数了，可以不重复定义，或者保持一致
        # 这里可以删掉或者保留，看你的需求

    def forward(self, x):
        # 构造假的 q_scale 列表，模拟训练态存在
        q_scale_enc = [torch.tensor(1.0)] * 64
        q_index = 1
        curr_q_enc = self.get_curr_q(q_scale_enc, self.q_basic_enc, q_index=q_index)
        # 调用正确的编码器模块 enc
        return self.enc(x, curr_q_enc)


# === 包装 P 帧模型 ===
class WrappedPFrameNet(DMC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加 dummy forward 仅供计算 FLOPs
        self.dummy = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.dummy(x)


def compute(model, name="Model"):
    model.eval()
    with torch.no_grad():
        try:
            macs, params = get_model_complexity_info(
                model,
                (3, 512, 512),
                as_strings=True,
                print_per_layer_stat=False,
                verbose=False,
            )
            print(f"{name}:\n  MACs: {macs}\n  Params: {params}")
        except RuntimeError as e:
            print(f"Error during FLOPs computation for {name}: {e}")



def forward(self, x):
    q_scale_enc = [torch.tensor(1.0)] * 64
    q_index = 1
    curr_q_enc = self.get_curr_q(q_scale_enc, self.q_basic_enc, q_index=q_index)
    return self.g_a(x, curr_q_enc)


if __name__ == "__main__":
    print("Estimating MACs and Params...\n")

    # I 帧模型
    i_model = WrappedIntraNet(ec_thread=False, stream_part=1, inplace=True)
    i_model.eval()
    compute(i_model, "I-frame model")

    # P 帧模型
    p_model = WrappedPFrameNet(ec_thread=False, stream_part=1, inplace=True)
    p_model.eval()
    compute(p_model, "P-frame model")
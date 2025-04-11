import torch
from models import RIN, MPN
from calflops import calculate_flops

# 이 코드는 RIN 모델만 측정한 것임

# class RINWrapper(torch.nn.Module):
#     def __init__(self, rin_model, mask, neck):
#         super().__init__()
#         self.rin = rin_model
#         self.mask = mask
#         self.neck = neck
#
#     def forward(self, x):
#         return self.rin(x, self.mask, self.neck)
#
#
# # 모델 초기화
# rin = RIN().cuda().eval()
# mpn = MPN().cuda().eval()
#
# # 입력 생성
# x = torch.randn(1, 3, 224, 224).cuda()
# mask = torch.randn(1, 1, 224, 224).cuda()
# _, neck = mpn(x)
#
# # Wrapper로 감싸기
# wrapped_rin = RINWrapper(rin, mask, neck).cuda().eval()
#
# # FLOPs 측정
# flops, macs, params = calculate_flops(
#     model=wrapped_rin,
#     input_shape=(1, 3, 224, 224),
#     output_as_string=True,
#     output_precision=4
# )
#
# print("RIN FLOPs: %s   MACs: %s   Params: %s\n" % (flops, macs, params))











# 이건 MPN까지 합쳐서 재는 코드

class FullInpaintingModel(torch.nn.Module):
    def __init__(self, mpn, rin, mask):
        super().__init__()
        self.mpn = mpn
        self.rin = rin
        self.mask = mask

    def forward(self, x):
        _, neck = self.mpn(x)
        return self.rin(x, self.mask, neck)


# 초기화
mpn = MPN().cuda().eval()
rin = RIN().cuda().eval()
mask = torch.randn(1, 1, 224, 224).cuda()

# 전체 모델 래핑
full_model = FullInpaintingModel(mpn, rin, mask)

# FLOPs 측정
flops, macs, params = calculate_flops(
    model=full_model,
    input_shape=(1, 3, 224, 224),
    output_as_string=True,
    output_precision=4
)

print("MPN + RIN FLOPs: %s   MACs: %s   Params: %s\n" % (flops, macs, params))

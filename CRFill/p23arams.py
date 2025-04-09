#
# import torch
# from convnext import GatedGenerator
#
# from calflops import calculate_flops
#
#
# print('Initializing model...')
#
# model = GatedGenerator().cuda()  # student model
# # model = UNet(n_channels=3, n_classes=3).cuda()
#
# # 모델을 평가 모드로 전환
# model.eval()
#
# # 입력 크기 정의 (배치 크기 포함하지 않음)
# img_size = (1, 4, 224, 224)  # 배치 크기 1, 채널 4, 224x224 이미지
#
#
# # FLOPs 계산
# flops, macs, params = calculate_flops(
#     model=model,
#     input_shape=img_size,  # 두 개의 입력 전달
#     output_as_string=True,
#     output_precision=4
# )
#
# print("GatedGenerator FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))



import torch
from models import TwostagendGenerator
from calflops import calculate_flops

print('Initializing model...')

# GatedGenerator의 forward에 기본 mask 추가
# 모델 래핑 (forward input: 4채널 = RGB+mask)
class WrappedGatedGenerator(TwostagendGenerator):
    def forward(self, x):
        image = x[:, :3, :, :]
        mask = x[:, 3:, :, :]
        return super().forward(image, mask)

# 모델 초기화
model = WrappedGatedGenerator().cuda()
model.eval()

# Dummy 입력 생성 (3채널 RGB + 1채널 mask)
input_size = (1, 4, 224, 224)

# FLOPs 계산
flops, macs, params = calculate_flops(
    model=model,
    input_shape=input_size,
    output_as_string=True,
    output_precision=4
)


print("GatedGenerator FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

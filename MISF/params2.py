 # 오류나면 kpn_module에서 128번 줄을
 #         kernels = F.interpolate(kernels,
 #                                 size=(256 * 9, data_with_est.shape[-1] // 4, data_with_est.shape[-2] // 4),
 #                                 mode='nearest') 로 바꾸고 할 것.
import torch
from models import InpaintGenerator

from calflops import calculate_flops


print('Initializing model...')

model = InpaintGenerator().cuda()  # student model
# model = UNet(n_channels=3, n_classes=3).cuda()

# 모델을 평가 모드로 전환
model.eval()

# 입력 크기 정의 (배치 크기 포함하지 않음)
inputsize = (1, 4, 224, 224)  # 배치 크기 1, 채널 4, 224x224 이미지


# FLOPs 계산
flops, macs, params = calculate_flops(
    model=model,
    input_shape=inputsize,  # 두 개의 입력 전달
    output_as_string=True,
    output_precision=4
)

print("GatedGenerator FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

# print(model)
# # 모델 요약
# summary(model, input_size=input_size)


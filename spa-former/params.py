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
from models import Generator
from calflops import calculate_flops

print('Initializing model...')

# GatedGenerator의 forward에 기본 mask 추가
class WrappedGatedGenerator(Generator):
    def forward(self, concated_img, mask=None):
        if mask is None:
            # 입력 크기와 동일한 크기의 기본 mask 생성
            mask = torch.ones_like(concated_img[:, :1, :, :])  # batch x 1 x H x W
        return super().forward(concated_img, mask)

# 모델 초기화
model = WrappedGatedGenerator().cuda()
model.eval()

# 입력 크기 정의
input_size = (1, 3, 224, 224)  # Batch size 포함되지 않음 (1은 임시 배치 크기)    # 실제로 모델에는 3채널, 1채널 각각 들어감 즉 forward에 (image, mask)로 돼있을때

# FLOPs 계산
flops, macs, params = calculate_flops(
    model=model,
    input_shape=input_size,
    output_as_string=True,
    output_precision=4
)

print("GatedGenerator FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

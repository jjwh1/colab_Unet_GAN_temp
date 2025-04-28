import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from KD_models_nodil_4B_1024_st1b import UNetGenerator_T                       # Teacher model
from KD_st_noMSA_1conv_lay123_4pool_st1b import UNetGenerator, Discriminator   # Student model
import adaptors                                                           # Adaptors
from dataset import InpaintDataset
from torchvision import transforms, utils
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
from datetime import datetime
import csv
from convnext import convnext_small  # [추가]

def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정
seed_everything(42)

# [추가] Feature L1 Loss 계산 함수
def feature_l1_loss(fake_images, gt_images, model, device):
    model.eval()
    with torch.no_grad():
        fake_features = model(fake_images.to(device))
        gt_features = model(gt_images.to(device))
    return nn.MSELoss()(fake_features, gt_features)


def train_gan_epoch(generator, generator_T, discriminator,adaptor_enc3,adaptor_bottleneck,adaptor_dec3,dataloader, criterion, optimizer_g, optimizer_d, device, recognition_model, lambda_adv=0.1):  # 한 에포크 학습 정의
    generator.train()
    generator_T.eval()
    discriminator.train()
    epoch_g_loss, epoch_g_l2_loss, epoch_g_adv_loss, epoch_d_loss, epoch_g_recog_loss, epoch_kd_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)


    for inputs, gts, masks, _ ,_ ,largemasks in progress_bar:
        batch_size = inputs.size(0)  # 현재 배치 크기
        total_samples += batch_size  # 전체 샘플 수 누적
        inputs, gts, masks, largemasks = inputs.to(device), gts.to(device), masks.to(device), largemasks.to(device)

        # Train Discriminator
        optimizer_d.zero_grad()
        fake_images, student_enc1,student_enc3,student_bottleneck,student_dec1,student_dec3 = generator(inputs)

        # fake_images, _,student_enc3,student_bottleneck,student_dec1,student_dec3 = generator(inputs)

        real_output = discriminator(gts) # dis의 output: (batch_size, 1) 형태의 출력 텐서
        # 네트워크에서 생성된 값이 아니라 데이터셋에서 직접 가져온 Ground Truth 이미지이기 때문에 이 데이터는 모델의 그래디언트 업데이트에 영향을 주는 학습 파라미터와 연결된 계산 그래프에 속하지 않음
        # 따라서, 이미 계산 그래프와 분리되어 있으므로 detach()가 필요하지 않습니다.
        # gts는 외부에서 불러온 이미지 (고정된 것이고, 변하면 안됨)이니 그라디언트 자체가 없음
        fake_output = discriminator(fake_images.detach()) # dis의 output: (batch_size, 1) 형태의 출력 텐서
        # 만약 detach()를 사용하지 않으면, fake_images를 통해 흘러간 그래디언트는 Generator까지 전파되어 Generator의 가중치가 갱신됨
        # 즉, fake_output이 d_loss에 반영되고, fake_output은 gen에서 만든 fake_images를 입력으로 받기 때문에 d_loss 최적화 시 fake_images에 영향을 주게 됨. 따라서 d_loss를 최적화하기 위해 fake_images가 그에 맞게 바뀔 수가 있음
        # fake_images가 그에 맞게 바뀐다는 말은 이걸 만든 generator가 바뀐다는거니 generator의 weight가 바뀌게 됨
        # gpt: fake_images.detach()는 Generator에서 생성된 이미지를 그래프에서 분리하기 위한 것입니다.
        d_loss_real = criterion(real_output, torch.ones_like(real_output).to(device))   # criterion이 BCE라고 하면 이미 배치 수만큼 평균 내주는게 내장돼있어 .mean()하지 않아도됨.
        d_loss_fake = criterion(fake_output, torch.zeros_like(fake_output).to(device))  # cross entropy는 0~1 확률값을 다룰 때 좋음. (동찬선배가 설명해준 그래프 생각)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()

        # ✅ Teacher 모델의 Output 및 Feature Map 가져오기 (Offline KD)
        with torch.no_grad():
            teacher_output, teacher_enc1,teacher_enc3,teacher_bottleneck,teacher_dec1,teacher_dec3 = generator_T(inputs)
            # teacher_output, _,teacher_enc3,teacher_bottleneck,teacher_dec1,teacher_dec3 = generator_T(inputs)

        fake_output = discriminator(fake_images)
        g_loss_adv = criterion(fake_output, torch.ones_like(fake_output).to(device))  # Discriminator를 잘 속이는지에 대한 지표(loss)
        # g_loss_pixel = nn.MSELoss()(fake_images, gts)  # Discriminator와 관계없이 gt image와 비교했을 때 잘 복원했는지에 대한 지표(loss)
        g_loss_pixel = nn.MSELoss()(fake_images * (1 - largemasks), gts * (1 - largemasks)) + 100 * nn.MSELoss()(fake_images * largemasks, gts * largemasks)
        # [추가] Feature L1 Loss 계산
        g_loss_recog = feature_l1_loss(fake_images, gts, recognition_model, device)

        # (4) KD Loss (Feature-Level & Output-Level)
        kd_loss_output = nn.MSELoss()(fake_images * (1 - largemasks), teacher_output * (1 - largemasks)) + 100 * nn.MSELoss()(fake_images * largemasks, teacher_output * largemasks)
        kd_loss_enc1 = nn.MSELoss()(student_enc1, teacher_enc1)
      
        kd_loss_enc3 = adaptor_enc3(student_enc3, teacher_enc3)
     
        kd_loss_bottleneck = adaptor_bottleneck(student_bottleneck, teacher_bottleneck)
        kd_loss_dec1 = nn.MSELoss()(student_dec1, teacher_dec1)
    
        kd_loss_dec3 = adaptor_dec3(student_dec3, teacher_dec3)
       
        

        total_kd_loss = 2 * kd_loss_output  +kd_loss_enc1+kd_loss_enc3+ kd_loss_bottleneck+kd_loss_dec1+kd_loss_dec3
        # total_kd_loss = kd_loss_enc1+kd_loss_enc2+kd_loss_enc3+kd_loss_enc4+kd_loss_bottleneck

        g_loss = g_loss_pixel + lambda_adv * g_loss_adv + lambda_adv *g_loss_recog + 5* total_kd_loss
        g_loss.backward()
        optimizer_g.step()

        epoch_g_loss += g_loss.item()* batch_size
        epoch_g_l2_loss += g_loss_pixel.item()* batch_size
        epoch_g_adv_loss += g_loss_adv.item()* batch_size
        epoch_g_recog_loss += g_loss_recog.item() * batch_size  # [추가]
        epoch_d_loss += d_loss.item()* batch_size
        epoch_kd_loss += total_kd_loss.item()* batch_size

        # Update progress bar with current losses
        progress_bar.set_postfix({"G_Loss": g_loss.item(), "G_L2_Loss": g_loss_pixel.item(), "G_adv_loss": g_loss_adv.item(), "G_recog_loss": g_loss_recog.item(),
                                  "D_Loss": d_loss.item(), "KD_Loss": total_kd_loss.item()})

    return (epoch_g_loss / total_samples,
            epoch_g_l2_loss / total_samples,
            epoch_g_adv_loss / total_samples,
            epoch_g_recog_loss / total_samples,
            epoch_d_loss / total_samples,
            epoch_kd_loss / total_samples)


def validate_epoch(generator, generator_T, discriminator,adaptor_enc3,adaptor_bottleneck,adaptor_dec3,dataloader, device, criterion, writer, recognition_model, lambda_adv, epoch,save_dir=None):
    generator.eval()
    generator_T.eval()
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    val_g_loss, val_g_l2_loss, val_g_adv_loss, val_g_recog_loss, val_kd_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    val_d_loss= 0.0
    total_samples = 0
    # Create a new directory for the epoch
    epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch + 1}')
    os.makedirs(epoch_save_dir, exist_ok=True)


    with torch.no_grad():
        for i, (inputs, gts, masks, filenames, _, largemasks) in enumerate(dataloader):
            batch_size = inputs.size(0)  # 현재 배치 크기
            total_samples += batch_size  # 전체 샘플 수 누적
            inputs, gts, masks, largemasks = inputs.to(device), gts.to(device), masks.to(device), largemasks.to(device)
            fake_images, student_enc1,student_enc3,student_bottleneck,student_dec1,student_dec3 = generator(inputs)       # student 호출
            teacher_output, teacher_enc1,teacher_enc3,teacher_bottleneck,teacher_dec1,teacher_dec3 = generator_T(inputs)  # teacher 호출

            # Save a few sample images

            if i < 6:  # Save up to 5 sample images per epoch
                # Convert from -1~1 to 0~1 for saving
                images = inputs[:, :3, :, :] # concat된 마스크 제외한 input 이미지만

                sample_images = fake_images.clamp(0, 1).cpu().numpy()  # Shape: (B, C, H, W)
                gt_images = gts.clamp(0, 1).cpu().numpy()
                input_images = images.clamp(0,1).cpu().numpy()  # Use only the first 3 channels for input

                # Loop through batch and save each image
                for idx in range(sample_images.shape[0]):
                    filename = filenames[idx]

                    # Reshape: (C, H, W) -> (H, W, C)
                    sample_image_np = (sample_images[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
                    gt_image_np = (gt_images[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
                    input_image_np = (input_images[idx].transpose(1, 2, 0) * 255).astype(np.uint8)

                    # Save images using OpenCV
                    cv2.imwrite(os.path.join(epoch_save_dir, f'sample_{i + 1}_{idx + 1}_{filename}.png'),
                                cv2.cvtColor(sample_image_np, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(epoch_save_dir, f'gt_{i + 1}_{idx + 1}_{filename}.png'),
                                cv2.cvtColor(gt_image_np, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(epoch_save_dir, f'input_{i + 1}_{idx + 1}_{filename}.png'),
                                cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR))

            g_loss_adv = criterion(discriminator(fake_images),
                                   torch.ones_like(discriminator(fake_images)).to(device))  # Discriminator를 잘 속이는지에 대한 지표(loss)
            # g_loss_pixel = nn.MSELoss()(fake_images, gts)  # Discriminator와 관계없이 gt image와 비교했을 때 잘 복원했는지에 대한 지표(loss)
            g_loss_pixel = nn.MSELoss()(fake_images * (1 - largemasks), gts * (1 - largemasks)) + 100 * nn.MSELoss()(fake_images * largemasks, gts * largemasks)

            g_loss_recog = feature_l1_loss(fake_images, gts, recognition_model, device)

            # ✅ KD Loss (Feature-Level & Output-Level)
            kd_loss_output = nn.MSELoss()(fake_images * (1 - largemasks), teacher_output * (1 - largemasks)) + 100 * nn.MSELoss()(fake_images * largemasks, teacher_output * largemasks)
            kd_loss_enc1 = nn.MSELoss()(student_enc1, teacher_enc1)
       
            kd_loss_enc3 = adaptor_enc3(student_enc3, teacher_enc3)
          
            kd_loss_bottleneck = adaptor_bottleneck(student_bottleneck, teacher_bottleneck)
            kd_loss_dec1 = nn.MSELoss()(student_dec1, teacher_dec1)
          
            kd_loss_dec3 = adaptor_dec3(student_dec3, teacher_dec3)
            
            
            total_kd_loss = 2 * kd_loss_output  +kd_loss_enc1+kd_loss_enc3+kd_loss_bottleneck+kd_loss_dec1+kd_loss_dec3
            # total_kd_loss = kd_loss_enc1+kd_loss_enc2+kd_loss_enc3+kd_loss_enc4+ kd_loss_bottleneck
            g_loss = g_loss_pixel + lambda_adv * g_loss_adv + lambda_adv * g_loss_recog + 5*total_kd_loss

            real_output = discriminator(gts)  # dis의 output: (batch_size, 1) 형태의 출력 텐서
            # 네트워크에서 생성된 값이 아니라 데이터셋에서 직접 가져온 Ground Truth 이미지이기 때문에 이 데이터는 모델의 그래디언트 업데이트에 영향을 주는 학습 파라미터와 연결된 계산 그래프에 속하지 않음
            # 따라서, 이미 계산 그래프와 분리되어 있으므로 detach()가 필요하지 않습니다.
            # gts는 외부에서 불러온 이미지 (고정된 것이고, 변하면 안됨)이니 그라디언트 자체가 없음
            fake_output = discriminator(fake_images.detach())  # dis의 output: (batch_size, 1) 형태의 출력 텐서
            # 만약 detach()를 사용하지 않으면, fake_images를 통해 흘러간 그래디언트는 Generator까지 전파되어 Generator의 가중치가 갱신됨
            # 즉, fake_output이 d_loss에 반영되고, fake_output은 gen에서 만든 fake_images를 입력으로 받기 때문에 d_loss 최적화 시 fake_images에 영향을 주게 됨. 따라서 d_loss를 최적화하기 위해 fake_images가 그에 맞게 바뀔 수가 있음
            # fake_images가 그에 맞게 바뀐다는 말은 이걸 만든 generator가 바뀐다는거니 generator의 weight가 바뀌게 됨
            # gpt: fake_images.detach()는 Generator에서 생성된 이미지를 그래프에서 분리하기 위한 것입니다.
            d_loss_real = criterion(real_output, torch.ones_like(real_output).to(
                device))  # criterion이 BCE라고 하면 이미 배치 수만큼 평균 내주는게 내장돼있어 .mean()하지 않아도됨.
            d_loss_fake = criterion(fake_output, torch.zeros_like(fake_output).to(
                device))  # cross entropy는 0~1 확률값을 다룰 때 좋음. (동찬선배가 설명해준 그래프 생각)
            d_loss = d_loss_real + d_loss_fake


            val_g_loss += g_loss.item()* batch_size
            val_g_l2_loss += g_loss_pixel.item()* batch_size
            val_g_adv_loss += g_loss_adv.item()* batch_size
            val_g_recog_loss += g_loss_recog.item() * batch_size  # [추가]
            val_d_loss += d_loss.item()* batch_size
            val_kd_loss += total_kd_loss.item()* batch_size
            psnr(fake_images, gts)  # 계속 누적됨 (각 배치당 psnr이 누적(배치size로 평균처리)돼서 한 epoch를 채우면 밑에서 compute를 통해 반환)
            ssim(fake_images, gts)  # 계속 누적됨 (각 배치당 ssim이 누적(배치size로 평균처리)돼서 한 epoch를 채우면 밑에서 compute를 통해 반환)

    val_g_loss /= total_samples
    val_g_l2_loss /= total_samples
    val_g_adv_loss /= total_samples
    val_g_recog_loss /= total_samples
    val_d_loss /= total_samples
    val_kd_loss /= total_samples
    psnr_value = psnr.compute().item()  # 명시적으로 값을 가져옴
    ssim_value = ssim.compute().item()

    writer.add_scalar("Validation/G_Loss", val_g_loss, epoch)
    writer.add_scalar("Validation/G_L2_Loss", val_g_l2_loss, epoch)
    writer.add_scalar("Validation/G_adv_Loss", val_g_adv_loss, epoch)
    writer.add_scalar("Validation/G_recog_Loss", val_g_recog_loss, epoch)
    writer.add_scalar("Validation/D_Loss", val_d_loss, epoch)
    writer.add_scalar("Validation/KD_Loss", val_kd_loss, epoch)
    writer.add_scalar("Validation/PSNR", psnr_value, epoch)
    writer.add_scalar("Validation/SSIM", ssim_value, epoch)


    return val_g_loss, val_g_l2_loss, val_g_adv_loss, val_g_recog_loss, val_d_loss, val_kd_loss, psnr_value, ssim_value

def load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator1_state_dict"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d1_state_dict"])
    epoch = checkpoint["epoch"]   # start_epoch 뱉을 때
    g_loss = checkpoint["g_loss"]
    g_l2_loss = checkpoint["g_l2_loss"]
    g_adv_loss = checkpoint["g_adv_loss"]
    d_loss = checkpoint["d_loss"]
    return generator, discriminator, optimizer_g, optimizer_d, epoch, g_loss, g_l2_loss, g_adv_loss, d_loss



def main():
    # Paths
    save_dir = "/content/drive/MyDrive/inpaint_result/UPOL/KD_SRD_Lendecoder_ADx_en1de1_L2_total_5__S_4pool_T_nodil4B1024_fold1_colab/db1_train"
    writer = SummaryWriter(os.path.join(save_dir, 'SR_Stage_4%s' % datetime.now().strftime("%Y%m%d-%H%M%S")))

    train_image_paths = '/content/dataset/UPOL/reflection_random(50to1.7)_db1_224_trainset'  # List of input image paths
    train_mask_paths = '/content/dataset/UPOL/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db1_test_layer12_0.3_only_mask_trainset'  # List of mask paths
    train_gt_paths = "/content/dataset/UPOL/db1_224_for_gt_inpainting_trainset"  # List of ground truth paths
    train_large_mask_paths = "/content/dataset/UPOL/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db1_test_layer12_0.3_only_mask_h3_w3.2_trainset"  # List of ground truth paths

    val_image_paths = '/content/dataset/UPOL/reflection_random(50to1.7)_db1_224_validset'  # List of input image paths
    val_mask_paths = '/content/dataset/UPOL/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db1_test_layer12_0.3_only_mask_validset'  # List of mask paths
    val_gt_paths = "/content/dataset/UPOL/db1_224_for_gt_inpainting_validset"  # List of ground truth paths
    val_large_mask_paths = "/content/dataset/UPOL/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db1_test_layer12_0.3_only_mask_h3_w3.2_validset"  # List of ground truth paths
    

    teacher_paths = '/content/dataset/T1_checkpoint_epoch.tar'
    

    results_path = os.path.join(save_dir, "metrics.csv")

    # [추가] Feature L1 Loss를 위한 ConvNeXt 모델 로드
    MODEL_PATH = "/content/dataset/recog1_saved_model_epoch.pth"

    os.makedirs(save_dir, exist_ok=True)

    # checkpoint_path = "D:/inpaint_result/CASIA_Distance/TT-Unet_GAN_D_100x100/db1_train_2/checkpoint_epoch_106.pth.tar"  # 불러올 시 마지막 저장된 pth파일 경로 입력!!
    checkpoint_path = None

    # Parameters
    batch_size = 8
    lr = 0.0002
    num_epochs = 400
    lambda_adv = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    recognition_model = convnext_small(pretrained=False)
    recognition_model.head = nn.Identity()  # Classification 헤드 제거
    recognition_model.load_state_dict(torch.load(MODEL_PATH), strict=False)
    recognition_model = recognition_model.to(device)
    recognition_model.eval()

    # Dataset and Dataloader
    train_dataset = InpaintDataset(train_image_paths, train_mask_paths, train_gt_paths, train_large_mask_paths)
    val_dataset = InpaintDataset(val_image_paths, val_mask_paths, val_gt_paths, val_large_mask_paths)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Models
    generator = UNetGenerator().to(device)
    generator_T = UNetGenerator_T().to(device)
    
    adaptor_enc3 = adaptors.SRD(128,256).to(device)
    adaptor_bottleneck = adaptors.SRD(256,1024).to(device)
    
    adaptor_dec3 = adaptors.SRD(128,256).to(device)
  
    

    checkpoint = torch.load(teacher_paths, map_location=device)
    generator_T.load_state_dict(checkpoint['generator_state_dict'])
    ''' teacher 모델 선학습 시 save할 때 model의 파라미터만 저장한게 아니고 optim, loss 등등 저장했기 때문에 
    단순 generator_T.load_state_dict(torch.load(teacher_paths, map_location=device))으로 하면 안됨  '''
    generator_T.eval()  # Teacher 모델은 학습하지 않음
    discriminator = Discriminator().to(device)

    # Optimizers
    # optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    # optimizer_g = optim.Adam(list(generator.parameters()) +
    # list(adaptor_enc3.parameters())+
    # list(adaptor_bottleneck.parameters())+
    # list(adaptor_dec3.parameters())
    # , lr=lr, betas=(0.5, 0.999))
    # optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    optimizer_g = optim.Adam(list(generator.parameters()) +
    list(adaptor_enc3.parameters())+
    list(adaptor_bottleneck.parameters())+
    list(adaptor_dec3.parameters())
    , lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # 그래프 local에 저장 위한
    g_losses = []  # train loss
    g_l2_losses = []  # train loss
    g_adv_losses = []  # train loss
    g_recog_losses = []  # [추가] Feature L1 Loss 저장
    d_losses = []  # train los
    kd_losses = []

    # Loss function
    criterion = nn.BCELoss()   # nn.BCELoss()는 nn.BCEWithLogitsLoss()와 달리 확률값을 입력으로 받음. discriminator가 sigmoid를 마지막에 거치므로 확률값이니 이걸 사용


    with open(results_path, mode='w', newline='') as csvfile:
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow([
            "epoch", "g_loss", "g_l2_loss", "g_adv_loss", "g_recog_loss", "d_loss", "kd_loss",
            "val_g_loss", "val_g_l2_loss", "val_g_adv_loss", "val_g_recog_loss", "val_d_loss", "val_kd_loss",
            "psnr", "ssim"
        ])


    start_epoch = 0

    if checkpoint_path:
        generator, discriminator, optimizer_g, optimizer_d, start_epoch, g_loss, g_l2_loss, g_adv_loss, d_loss \
            = load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d)


        print(f"Resuming training from epoch {start_epoch + 1}")


    for epoch in range(start_epoch, num_epochs):

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        g_loss, g_l2_loss, g_adv_loss, g_recog_loss, d_loss, kd_loss = train_gan_epoch(generator, generator_T, discriminator,adaptor_enc3,adaptor_bottleneck 
                                                                                           ,adaptor_dec3
                                                                                       , train_loader, criterion, optimizer_g, optimizer_d, device, recognition_model, lambda_adv)
        g_losses.append(g_loss)
        g_l2_losses.append(g_l2_loss)
        g_adv_losses.append(g_adv_loss)
        g_recog_losses.append(g_recog_loss)
        d_losses.append(d_loss)
        kd_losses.append(kd_loss)

        # Validation metrics and sample image saving
        val_g_loss, val_g_l2_loss, val_g_adv_loss, val_g_recog_loss, val_d_loss, val_kd_loss, psnr, ssim = validate_epoch(generator,generator_T,discriminator,adaptor_enc3,adaptor_bottleneck
                                                                                                                          ,adaptor_dec3,val_loader, device, criterion, writer, recognition_model, lambda_adv, epoch, save_dir)


        writer.add_scalar("Train/G_Loss", g_loss, epoch)
        writer.add_scalar("Train/G_L2", g_l2_loss, epoch)
        writer.add_scalar("Train/G_adv", g_adv_loss, epoch)
        writer.add_scalar("Train/G_recog", g_recog_loss, epoch)
        writer.add_scalar("Train/D_Loss", d_loss, epoch)
        writer.add_scalar("Train/KD_Loss", kd_loss, epoch)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], G_Loss: {g_loss:.4f}, g_l2_Loss: {g_l2_loss:.4f}, g_adv_Loss: {g_adv_loss:.4f}, g_recog_Loss: {g_recog_loss:.4f} d_Loss: {d_loss:.4f}, kd_Loss: {kd_loss:.4f}")
        print("----------validation-----------")
        print(
            f"val_g_loss: {val_g_loss:.4f}, val_g_l2_loss: {val_g_l2_loss:.4f} ,val_g_adv_loss: {val_g_adv_loss:.4f}, val_g_recog_loss: {val_g_recog_loss:.4f}, val_d_loss: {val_d_loss:.4f}"
            f",val_kd_loss: {val_kd_loss:.4f} ,PSNR: {psnr:.4f}, SSIM: {ssim:.4f} ")

        with open(results_path, mode='a', newline='') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow([
                epoch + 1, g_loss, g_l2_loss, g_adv_loss, g_recog_loss, d_loss, kd_loss,
                val_g_loss, val_g_l2_loss, val_g_adv_loss, val_g_recog_loss, val_d_loss, val_kd_loss,
                psnr, ssim
            ])

        if epoch >= 100:
            # Save checkpoint
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "g_loss": g_loss,
                "g_l2_loss": g_l2_loss,
                "g_adv_loss": g_adv_loss,
                "d_loss": d_loss,

                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict()

            }, os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.tar"))

    writer.close()


if __name__ == "__main__":
    main()
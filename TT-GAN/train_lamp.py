import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models import GatedGenerator, PatchDiscriminator1, PatchDiscriminator2
from dataset import InpaintDataset
from torchvision import transforms, utils
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from pytorch_fid import fid_score
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
from datetime import datetime
import cv2
import torch.nn.functional as F
import csv  # ### 수정됨

# https://github.com/csqiangwen/DeepFillv2_Pytorch/blob/master/network.py  gated convolution 코드 참고
# https://github.com/sirius-image-inpainting/Free-Form-Image-Inpainting-With-Gated-Convolution/blob/main/model/loss.py#L28 참고

def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    # torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정     -> False로 하거나 아예 없애면 빨라짐 (재현성은 불가능)
    # torch.backends.cudnn.benchmark = False                                               -> True로 하거나 아예 없애면 빨라짐 (재현성은 불가능)
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정
seed_everything(42)


# 이게 training ratio 3:1 지킨 버전
def train_gan_epoch(generator, discriminator1, discriminator2, dataloader, optimizer_g, optimizer_d1, optimizer_d2,device):  # 한 에포크 학습 정의
    generator.train()
    discriminator1.train()
    discriminator2.train()
    epoch_g_loss, epoch_g1_l1_loss, epoch_g2_l1_loss, epoch_d1_loss, epoch_d2_loss = 0.0, 0.0,0.0,0.0,0.0
    epoch_g1_adv_loss, epoch_g2_adv_loss = 0.0,0.0

    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)


    for batch_idx, (images, gts, masks, large_masks) in enumerate(progress_bar):  # 한 epoch 당 과정
        images, gts, masks, large_masks = images.to(device), gts.to(device), masks.to(device), large_masks.to(device) # inputs = 4채널 concat된 이미지
        batch_size = images.size(0)  # 현재 배치 크기
        total_samples += batch_size  # 전체 샘플 수 누적
        fake_images1, fake_images2 = generator(images, masks, large_masks)

        # Step 1: Train Discriminators
        if batch_idx % 3 == 0:  # Discriminators are trained every 3 batches (D먼저 학습하는거긴함)


            # Train Discriminator1
            optimizer_d1.zero_grad()
            real_output1 = discriminator1(gts)
            fake_output1 = discriminator1(fake_images2.detach())
            # d1_loss_real = -torch.mean(
            #     torch.min(torch.zeros_like(real_output1).to(device),
            #               -torch.ones_like(real_output1).to(device) + real_output1)
            # )
            d1_loss_real = torch.mean(F.relu(1. - real_output1))
            # d1_loss_fake = -torch.mean(
            #     torch.min(torch.zeros_like(fake_output1).to(device),
            #               -torch.ones_like(fake_output1).to(device) - fake_output1)
            # )
            d1_loss_fake = torch.mean(F.relu(1. + fake_output1))
            d1_loss = d1_loss_real + d1_loss_fake

            d1_loss.backward()
            optimizer_d1.step()

            # Train Discriminator2
            optimizer_d2.zero_grad()

            fake_output2 = discriminator2(fake_images2.detach(), large_masks)
            real_output2 = discriminator2(gts, large_masks)

            # d2_loss_real = -torch.mean(
            #     torch.min(torch.zeros_like(real_output2).to(device),
            #               -torch.ones_like(real_output2).to(device) + real_output2)
            # )
            d2_loss_real = torch.mean(F.relu(1. - real_output2))
            # d2_loss_fake = -torch.mean(
            #     torch.min(torch.zeros_like(fake_output2).to(device),
            #               -torch.ones_like(fake_output2).to(device) - fake_output2)
            # )
            d2_loss_fake = torch.mean(F.relu(1. + fake_output2))
            d2_loss = d2_loss_real + d2_loss_fake
            d2_loss.backward()
            optimizer_d2.step()

            epoch_d1_loss += d1_loss.item()* batch_size
            epoch_d2_loss += d2_loss.item()* batch_size

        # Step 2: Train Generator

        optimizer_g.zero_grad()
        g1_l1_loss = nn.L1Loss()(fake_images1, gts)  # G1의 L1 Loss
        g1_adv_loss = -torch.mean(discriminator1(fake_images2))  # G1의 Unet_GAN_D_100x100 Loss
        g2_l1_loss = nn.L1Loss()(fake_images2, gts)  # G2의 L1 Loss
        g2_adv_loss = -torch.mean(discriminator2(fake_images2, large_masks))  # G2의 Unet_GAN_D_100x100 Loss
        g_loss = g1_l1_loss + g2_l1_loss + g1_adv_loss + g2_adv_loss
        g_loss.backward()
        optimizer_g.step()

        epoch_g_loss += g_loss.item()* batch_size
        epoch_g1_l1_loss += g1_l1_loss.item()* batch_size
        epoch_g2_l1_loss += g2_l1_loss.item()* batch_size
        epoch_g1_adv_loss += g1_adv_loss.item()* batch_size
        epoch_g2_adv_loss += g2_adv_loss.item()* batch_size

        # 진행 상황 업데이트
        progress_bar.set_postfix(
            {
                "G_Loss": g_loss.item(),
                "G1_L1": g1_l1_loss.item(),
                "G2_L1": g2_l1_loss.item(),
                "G1_Adv": g1_adv_loss.item(),
                "G2_Adv": g2_adv_loss.item(),
                "D1_Loss": d1_loss.item() if batch_idx % 3 == 0 else 0,
                "D2_Loss": d2_loss.item() if batch_idx % 3 == 0 else 0,
            }
        )

    return (epoch_g_loss / total_samples, epoch_g1_l1_loss/ total_samples, epoch_g2_l1_loss / total_samples,
            epoch_g1_adv_loss / total_samples,
            epoch_g2_adv_loss / total_samples, epoch_d1_loss / (total_samples//3), epoch_d2_loss / (total_samples//3))

# def validate_epoch(generator, dataloader, device, save_dir=None, epoch=None):
def validate_epoch(generator, discriminator1, discriminator2, dataloader, device, writer, epoch, save_dir=None):
    generator.eval()
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    val_g_loss, val_g1_l1_loss, val_g2_l1_loss = 0.0, 0.0, 0.0
    val_g1_adv_loss, val_g2_adv_loss = 0.0, 0.0
    val_d1_loss, val_d2_loss = 0.0, 0.0

    total_samples = 0

    # Create a new directory for the epoch
    epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch + 1}')
    os.makedirs(epoch_save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, gts, masks, large_masks) in enumerate(dataloader):  # 인덱스까지 얻기 위해 굳이 enumerate 사용 (i를 통해 5개만 저장하려고)

            images, gts, masks, large_masks = images.to(device), gts.to(device), masks.to(device), large_masks.to(
                device)  # inputs = 4채널 concat된 이미지

            batch_size = images.size(0)  # 현재 배치 크기
            total_samples += batch_size  # 전체 샘플 수 누적

            fake_images1, fake_images2 = generator(images, masks, large_masks)

            # Save a few sample images (정규화를 0~1 로 했던 경우)
            # if i < 5:  # Save up to 5 sample images per epoch
            #     utils.save_image(fake_images2, os.path.join(epoch_save_dir, f'sample_{i + 1}.png'))
            #     utils.save_image(gts, os.path.join(epoch_save_dir, f'gt_{i + 1}.png'))
            #     utils.save_image(images, os.path.join(epoch_save_dir, f'input_{i + 1}.png'))

            # Save a few sample images (정규화를 -1~1 로 했던 경우)
            if i < 5:  # Save up to 5 sample images per epoch
                # Convert from -1~1 to 0~1 for saving

                sample_images = ((fake_images2 + 1) / 2).clamp(-1, 1).cpu().numpy()  # Shape: (B, C, H, W)
                gt_images = ((gts + 1) / 2).clamp(-1, 1).cpu().numpy()
                input_images = ((images + 1) / 2).clamp(-1,1).cpu().numpy()  # Use only the first 3 channels for input

                # Loop through batch and save each image
                for idx in range(sample_images.shape[0]):
                    # Reshape: (C, H, W) -> (H, W, C)
                    sample_image_np = (sample_images[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
                    gt_image_np = (gt_images[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
                    input_image_np = (input_images[idx].transpose(1, 2, 0) * 255).astype(np.uint8)

                    # Save images using OpenCV
                    cv2.imwrite(os.path.join(epoch_save_dir, f'sample_{i + 1}_{idx + 1}.png'),
                                cv2.cvtColor(sample_image_np, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(epoch_save_dir, f'gt_{i + 1}_{idx + 1}.png'),
                                cv2.cvtColor(gt_image_np, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(epoch_save_dir, f'input_{i + 1}_{idx + 1}.png'),
                                cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR))

                # utils.save_image(sample_images, os.path.join(epoch_save_dir, f'sample_{i + 1}.png'))
                # utils.save_image(gt_images, os.path.join(epoch_save_dir, f'gt_{i + 1}.png'))
                # utils.save_image(input_images, os.path.join(epoch_save_dir, f'input_{i + 1}.png'))

            # Compute validation losses
            # G1 losses
            g1_l1_loss = nn.L1Loss()(fake_images1, gts)
            g1_adv_loss = -torch.mean(discriminator1(fake_images2))

            # G2 losses
            g2_l1_loss = nn.L1Loss()(fake_images2, gts)
            g2_adv_loss = -torch.mean(discriminator2(fake_images2, large_masks))

            # Total generator loss
            g_loss = g1_l1_loss + g2_l1_loss + g1_adv_loss + g2_adv_loss

            # D1 losses
            real_output1 = discriminator1(gts)
            fake_output1 = discriminator1(fake_images2.detach())
            # d1_loss_real = -torch.mean(
            #     torch.min(torch.zeros_like(real_output1).to(device),
            #               -torch.ones_like(real_output1).to(device) + real_output1)
            # )
            d1_loss_real = torch.mean(F.relu(1. - real_output1))
            # d1_loss_fake = -torch.mean(
            #     torch.min(torch.zeros_like(fake_output1).to(device),
            #               -torch.ones_like(fake_output1).to(device) - fake_output1)
            # )
            d1_loss_fake = torch.mean(F.relu(1. + fake_output1))
            d1_loss = d1_loss_real + d1_loss_fake

            # D2 losses
            real_output2 = discriminator2(gts, large_masks)
            fake_output2 = discriminator2(fake_images2.detach(), large_masks)
            # d2_loss_real = -torch.mean(
            #     torch.min(torch.zeros_like(real_output2).to(device),
            #               -torch.ones_like(real_output2).to(device) + real_output2)
            # )
            d2_loss_real = torch.mean(F.relu(1. - real_output2))
            # d2_loss_fake = -torch.mean(
            #     torch.min(torch.zeros_like(fake_output2).to(device),
            #               -torch.ones_like(fake_output2).to(device) - fake_output2)
            # )
            d2_loss_fake = torch.mean(F.relu(1. + fake_output2))
            d2_loss = d2_loss_real + d2_loss_fake

            # Aggregate losses
            val_g_loss += g_loss.item()* batch_size
            val_g1_l1_loss += g1_l1_loss.item()* batch_size
            val_g2_l1_loss += g2_l1_loss.item()* batch_size
            val_g1_adv_loss += g1_adv_loss.item()* batch_size
            val_g2_adv_loss += g2_adv_loss.item()* batch_size
            val_d1_loss += d1_loss.item()* batch_size
            val_d2_loss += d2_loss.item()* batch_size

            psnr(fake_images2, gts)
            ssim(fake_images2, gts)

            # Compute averages
    val_g_loss /= total_samples
    val_g1_l1_loss /= total_samples
    val_g2_l1_loss /= total_samples
    val_g1_adv_loss /= total_samples
    val_g2_adv_loss /= total_samples
    val_d1_loss /= total_samples
    val_d2_loss /= total_samples
    psnr_value = psnr.compute().item()  # 명시적으로 값을 가져옴
    ssim_value = ssim.compute().item()

        # Add to TensorBoard
    writer.add_scalar("Validation/G_Loss", val_g_loss, epoch)
    writer.add_scalar("Validation/G1_L1", val_g1_l1_loss, epoch)
    writer.add_scalar("Validation/G2_L1", val_g2_l1_loss, epoch)
    writer.add_scalar("Validation/G1_Adv", val_g1_adv_loss, epoch)
    writer.add_scalar("Validation/G2_Adv", val_g2_adv_loss, epoch)
    writer.add_scalar("Validation/D1_Loss", val_d1_loss, epoch)
    writer.add_scalar("Validation/D2_Loss", val_d2_loss, epoch)
    writer.add_scalar("Validation/PSNR", psnr_value, epoch)  # 수정
    writer.add_scalar("Validation/SSIM", ssim_value, epoch)  # 수정


    return val_g_loss, val_g1_l1_loss, val_g2_l1_loss, val_g1_adv_loss, val_g2_adv_loss, val_d1_loss, val_d2_loss, psnr_value, ssim_value
    #         val_loss += nn.L1Loss()(fake_images, gts).item()  # Validation loss 계산
    #
    #         psnr(fake_images, gts) # 계속 누적됨 (각 배치당 psnr이 누적(배치size로 평균처리)돼서 한 epoch를 채우면 밑에서 compute를 통해 반환)
    #         ssim(fake_images, gts) # 계속 누적됨 (각 배치당 ssim이 누적(배치size로 평균처리)돼서 한 epoch를 채우면 밑에서 compute를 통해 반환)
    #
    # val_loss /= len(dataloader)  # 평균 Validation loss
    #
    # return val_loss, psnr.compute().item(), ssim.compute().item()  # .compute()를 통해 누적된 값에 대한 최종 결과(평균내서) 반환



def save_loss_graph(losses, labels, save_dir, filename):
    plt.figure(figsize=(10, 5))
    for loss, label in zip(losses, labels):
        plt.plot(loss, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def load_checkpoint(checkpoint_path, generator, discriminator1, discriminator2, optimizer_g, optimizer_d1, optimizer_d2):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator1.load_state_dict(checkpoint["discriminator1_state_dict"])
    discriminator2.load_state_dict(checkpoint["discriminator2_state_dict"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    optimizer_d1.load_state_dict(checkpoint["optimizer_d1_state_dict"])
    optimizer_d2.load_state_dict(checkpoint["optimizer_d2_state_dict"])
    epoch = checkpoint["epoch"]   # start_epoch 뱉을 때
    g_loss = checkpoint["g_loss"]
    g1_l1_loss = checkpoint["g1_l1_loss"]
    g2_l1_loss = checkpoint["g2_l1_loss"]
    g1_adv_loss = checkpoint["g1_adv_loss"]
    g2_adv_loss = checkpoint["g2_adv_loss"]
    d1_loss = checkpoint["d1_loss"]
    d2_loss = checkpoint["d2_loss"]
    return generator, discriminator1, discriminator2, optimizer_g, optimizer_d1, optimizer_d2, epoch, g_loss, g1_l1_loss, g2_l1_loss, g1_adv_loss, g2_adv_loss, d1_loss, d2_loss



def main():
    save_dir = r"D:\inpaint_result\CASIA_Lamp\TT_GAN_fold2_py_\db2_train"

    writer = SummaryWriter(os.path.join(save_dir, 'SR_Stage_4%s' % datetime.now().strftime("%Y%m%d-%H%M%S")))
    # Paths
    train_image_paths = r"C:\Users\8138\Desktop\DB\CASIAv4\CASIA_Iris_Lamp\reflection_random(50to1.7)_db2_224_trainset"  # List of input image paths
    train_mask_paths = r"D:\mask\CASIA_Lamp\algorithm\450to50000_174x174padding_if_gac1_4000_algorithm\db2_test_layer12_0.3_only_mask_trainset"  # List of mask paths
    train_gt_paths = r"C:\Users\8138\Desktop\DB\CASIAv4\CASIA_Iris_Lamp\db2_224_for_gt_inpainting_trainset"  # List of ground truth paths
    train_large_mask_dir = r"D:\mask\CASIA_Lamp\algorithm\450to50000_174x174padding_if_gac1_4000_algorithm\db2_test_layer12_0.3_only_mask_h2.8_w3_trainset"  # List of large_mask paths

    val_image_paths = r"C:\Users\8138\Desktop\DB\CASIAv4\CASIA_Iris_Lamp\reflection_random(50to1.7)_db2_224_validset"  # List of input image paths
    val_mask_paths = r"D:\mask\CASIA_Lamp\algorithm\450to50000_174x174padding_if_gac1_4000_algorithm\db2_test_layer12_0.3_only_mask_validset"  # List of mask paths
    val_gt_paths = r"C:\Users\8138\Desktop\DB\CASIAv4\CASIA_Iris_Lamp\db2_224_for_gt_inpainting_validset"  # List of ground truth paths
    val_large_mask_dir = r"D:\mask\CASIA_Lamp\algorithm\450to50000_174x174padding_if_gac1_4000_algorithm\db2_test_layer12_0.3_only_mask_h2.8_w3_validset"  # List of large_mask paths
    os.makedirs(save_dir, exist_ok=True)

    results_path = os.path.join(save_dir, "metrics.csv")

    # checkpoint_path = "D:/inpaint_result/CASIA_Distance/TT-Unet_GAN_D_100x100/db1_train_2/checkpoint_epoch_106.pth.tar"  # 불러올 시 마지막 저장된 pth파일 경로 입력!!
    checkpoint_path = None

    # Parameters
    batch_size = 8
    lr_g = 0.0001
    lr_d1 = 0.000005
    lr_d2 = 0.00001
    num_epochs = 400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Dataset and Dataloader
    train_dataset = InpaintDataset(train_image_paths, train_mask_paths, train_gt_paths, train_large_mask_dir)
    val_dataset = InpaintDataset(val_image_paths, val_mask_paths, val_gt_paths, val_large_mask_dir)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Models
    generator = GatedGenerator().to(device)
    discriminator1 = PatchDiscriminator1().to(device)
    discriminator2 = PatchDiscriminator2().to(device)

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d1 = optim.Adam(discriminator1.parameters(), lr=lr_d1, betas=(0.5, 0.999))
    optimizer_d2 = optim.Adam(discriminator2.parameters(), lr=lr_d2, betas=(0.5, 0.999))



    # 그래프 local에 저장 위한
    g_losses = []   # train loss
    g1_l1_losses = []   # train loss
    g2_l1_losses = []   # train loss
    g1_adv_losses = []   # train loss
    g2_adv_losses = []   # train loss

    d1_losses = []   # train loss
    d2_losses = []   # train loss


    with open(results_path, mode="w", newline="") as file:  # CSV 파일 초기화
        writer_csv = csv.writer(file)
        writer_csv.writerow([
            "epoch", "g_loss", "g1_l1_loss", "g2_l1_loss", "g1_adv_loss", "g2_adv_loss",
            "d1_loss", "d2_loss", "val_g_loss", "val_g1_l1_loss", "val_g2_l1_loss",
            "val_g1_adv_loss", "val_g2_adv_loss", "val_d1_loss", "val_d2_loss", "psnr", "ssim"
        ])

    start_epoch = 0

    if checkpoint_path:
        generator, discriminator1, discriminator2, optimizer_g, optimizer_d1, optimizer_d2, start_epoch, g_loss, g1_l1_loss, g2_l1_loss, g1_adv_loss, g2_adv_loss, d1_loss, d2_loss\
            = load_checkpoint(checkpoint_path, generator, discriminator1, discriminator2, optimizer_g, optimizer_d1, optimizer_d2)
        print(f"Resuming training from epoch {start_epoch + 1}")


    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        g_loss, g1_l1_loss, g2_l1_loss, g1_adv_loss, g2_adv_loss, d1_loss, d2_loss = train_gan_epoch(generator, discriminator1, discriminator2, train_loader,
                                                                                                     optimizer_g, optimizer_d1, optimizer_d2, device)
        g_losses.append(g_loss)
        g1_l1_losses.append(g1_l1_loss)
        g2_l1_losses.append(g2_l1_loss)
        g1_adv_losses.append(g1_adv_loss)
        g2_adv_losses.append(g2_adv_loss)

        d1_losses.append(d1_loss)
        d2_losses.append(d2_loss)


        # Validation metrics and sample image saving

        # val_loss, psnr, ssim = validate_epoch(generator, val_loader, device, save_dir, epoch)
        # val_losses.append(val_loss)  # Validation loss 저장

        val_g_loss, val_g1_l1_loss, val_g2_l1_loss, val_g1_adv_loss, val_g2_adv_loss, val_d1_loss, val_d2_loss, psnr, ssim = validate_epoch(
            generator, discriminator1, discriminator2, val_loader, device, writer, epoch, save_dir
        )



        writer.add_scalar("Train/G_Loss", g_loss, epoch)
        writer.add_scalar("Train/G1_L1", g1_l1_loss, epoch)
        writer.add_scalar("Train/G2_L1", g2_l1_loss, epoch)
        writer.add_scalar("Train/G1_Adv", g1_adv_loss, epoch)
        writer.add_scalar("Train/G2_Adv", g2_adv_loss, epoch)
        writer.add_scalar("Train/D1_Loss", d1_loss, epoch)
        writer.add_scalar("Train/D2_Loss", d2_loss, epoch)

        with open(results_path, mode="a", newline="") as file:  # 매 에포크마다 결과 추가
            writer_csv = csv.writer(file)
            writer_csv.writerow([
                epoch + 1, g_loss, g1_l1_loss, g2_l1_loss, g1_adv_loss, g2_adv_loss,
                d1_loss, d2_loss, val_g_loss, val_g1_l1_loss, val_g2_l1_loss,
                val_g1_adv_loss, val_g2_adv_loss, val_d1_loss, val_d2_loss, psnr, ssim
            ])

        print(f"Epoch [{epoch + 1}/{num_epochs}], G_Loss: {g_loss:.4f}, D1_Loss: {d1_loss:.4f}, D2_Loss: {d2_loss:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
        print(f"g1_l1_loss: {g1_l1_loss:.4f}, g2_l1_loss: {g2_l1_loss:.4f}, ,g1_adv_loss: {g1_adv_loss:.4f}, g2_adv_loss: {g2_adv_loss:.4f} ")
        print(f"val_g_loss: {val_g_loss:.4f}, val_g1_l1_loss: {val_g1_l1_loss:.4f}, ,val_g1_adv_loss: {val_g1_adv_loss:.4f}, val_g2_l1_loss: {val_g2_l1_loss:.4f}, val_g2_adv_loss: {val_g2_adv_loss:.4f}, ")
        print(f"val_d1_loss: {val_d1_loss:.4f}, val_d2_loss: {val_d2_loss:.4f}, ,val_g1_adv_loss: {val_g1_adv_loss:.4f}, val_g2_l1_loss: {val_g2_l1_loss:.4f}, val_g2_adv_loss: {val_g2_adv_loss:.4f}, ")
        if epoch >= 150:
            # Save checkpoint
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "g_loss": g_loss,
                "g1_l1_loss": g1_l1_loss,
                "g2_l1_loss": g2_l1_loss,
                "g1_adv_loss": g1_adv_loss,
                "g2_adv_loss": g2_adv_loss,
                "d1_loss": d1_loss,
                "d2_loss": d2_loss,
                "discriminator1_state_dict": discriminator1.state_dict(),
                "discriminator2_state_dict": discriminator2.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d1_state_dict": optimizer_d1.state_dict(),
                "optimizer_d2_state_dict": optimizer_d2.state_dict()
            }, os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth.tar"))

    writer.close()

    save_loss_graph(
        [g_losses, g1_l1_losses, g2_l1_losses, g1_adv_losses, g2_adv_losses, d1_losses, d2_losses, val_g_loss, val_g1_l1_loss, val_g2_l1_loss, val_g1_adv_loss, val_g2_adv_loss, val_d1_loss, val_d2_loss],
        ["G Loss", "G1 L1 Loss", "G2 L1 Loss", "G1 Adv Loss", "G2 Adv Loss", "D1 Loss", "D2 Loss", "val_g_loss", "val_g1_l1_loss", "val_g2_l1_loss", "val_g1_adv_loss", "val_g2_adv_loss", "val_d1_loss", "val_d2_loss"],
        save_dir,
        "all_losses.png",
    )




if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models import MPN, RIN, Discriminator, PatchDiscriminator
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
import torch.nn.functional as F
from loss import IDMRFLoss,SemanticConsistencyLoss, compute_gradient_penalty, WeightedBCELoss
import kornia

def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    # torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정
seed_everything(42)


def train_gan_epoch(generator, mpn,discriminator, patchdiscriminator,dataloader, optimizer_g,optimizer_d, weighted_bce_loss, semantic_consistency_loss, texture_consistency_loss,device):  # 한 에포크 학습 정의
    generator.train()
    mpn.train()
    discriminator.train()
    patchdiscriminator.train()
    epoch_g_loss, epoch_g_l2_loss, epoch_g_adv_loss, epoch_d_loss = 0.0, 0.0, 0.0, 0.0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, gts, masks,largemasks in progress_bar:
        batch_size = images.size(0)  # 현재 배치 크기
        total_samples += batch_size  # 전체 샘플 수 누적
        images, gts, masks, largemasks = images.to(device), gts.to(device), masks.to(device), largemasks.to(device)

        batch_size, channels, h, w = images.size()
        unknown_pixel_ratio = torch.sum(masks.view(batch_size, -1), dim=1).mean() / (h * w)


        optimizer_d.zero_grad()
        pred_masks, neck = mpn(images)
        fake_images = generator(images,pred_masks, neck)

        real_global_validity = discriminator(gts).mean()
        fake_global_validity = discriminator(fake_images.detach()).mean()
        gp_global = compute_gradient_penalty(discriminator, fake_images.data, gts.data)

        real_patch_validity = patchdiscriminator(gts, masks).mean()
        fake_patch_validity = patchdiscriminator(fake_images.detach(), masks).mean()
        gp_fake = compute_gradient_penalty(patchdiscriminator, fake_images.data, gts.data, masks)

        real_validity = real_global_validity + real_patch_validity
        fake_validity = fake_global_validity + fake_patch_validity
        gp = gp_global + gp_fake

        d_loss = -real_validity + fake_validity + 10* gp
        #d_loss = -real_validity + fake_validity

        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()

        pred_masks, neck = mpn(images)

        m_loss = weighted_bce_loss(pred_masks, masks,
                                        torch.tensor([1 - unknown_pixel_ratio, unknown_pixel_ratio]))
        # m_loss = nn.BCELoss()(pred_masks, masks)


        fake_images = generator(images, pred_masks.detach(), neck.detach())
        g_loss_pixel = nn.L1Loss()(fake_images,gts)
        # sem_const_loss = semantic_consistency_loss(fake_images, gts)
        # tex_const_loss = texture_consistency_loss(fake_images, gts)
        adv_global_loss = -discriminator(fake_images).mean()
        adv_patch_loss = -patchdiscriminator(fake_images, masks).mean()
        g_loss_adv = adv_global_loss + adv_patch_loss

        # g_loss = 1.4* g_loss_pixel + 0.0001* sem_const_loss + 0.001* tex_const_loss  +0.001 * g_loss_adv
        g_loss = 0.8* g_loss_pixel + 0.001 * g_loss_adv

        (g_loss+m_loss).backward()
        optimizer_g.step()

        epoch_g_loss += g_loss.item()* batch_size
        epoch_g_l2_loss += g_loss_pixel.item()* batch_size
        epoch_g_adv_loss += g_loss_adv.item()* batch_size
        epoch_d_loss += d_loss.item()* batch_size

        # Update progress bar with current losses
        progress_bar.set_postfix({"G_Loss": g_loss.item(), "G_L2_Loss": g_loss_pixel.item(), "G_adv_loss": g_loss_adv.item(), "D_Loss": d_loss.item()})

    return (epoch_g_loss / total_samples,
            epoch_g_l2_loss / total_samples,
            epoch_g_adv_loss / total_samples,
            epoch_d_loss / total_samples)


def validate_epoch(generator,mpn, discriminator, dataloader, device, writer, epoch,save_dir=None):
    generator.eval()
    mpn.eval()
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    val_g_loss, val_g_l2_loss, val_g_adv_loss = 0.0, 0.0, 0.0
    val_d_loss= 0.0
    total_samples = 0
    # Create a new directory for the epoch
    epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch + 1}')
    os.makedirs(epoch_save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, gts, masks, largemasks) in enumerate(dataloader):
            batch_size = images.size(0)  # 현재 배치 크기
            total_samples += batch_size  # 전체 샘플 수 누적
            images, gts, masks, largemasks = images.to(device), gts.to(device), masks.to(device), largemasks.to(device)

            pred_masks, neck = mpn(images)
            fake_images = generator(images,pred_masks, neck)

            # Save a few sample images

            if i < 6:  # Save up to 5 sample images per epoch
                # Convert from -1~1 to 0~1 for saving
                images = images[:, :3, :, :] # concat된 마스크 제외한 input 이미지만

                # sample_images = fake_images.clamp(0, 1).cpu().numpy()  # Shape: (B, C, H, W)
                # gt_images = gts.clamp(0, 1).cpu().numpy()
                # input_images = images.clamp(0,1).cpu().numpy()  # Use only the first 3 channels for input

                sample_images = ((fake_images + 1) / 2).clamp(-1, 1).cpu().numpy()  # Shape: (B, C, H, W)
                gt_images = ((gts + 1) / 2).clamp(-1, 1).cpu().numpy()
                input_images = ((images + 1) / 2).clamp(-1, 1).cpu().numpy()  # Use only the first 3 channels for input

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


            psnr(fake_images, gts)  # 계속 누적됨 (각 배치당 psnr이 누적(배치size로 평균처리)돼서 한 epoch를 채우면 밑에서 compute를 통해 반환)
            ssim(fake_images, gts)  # 계속 누적됨 (각 배치당 ssim이 누적(배치size로 평균처리)돼서 한 epoch를 채우면 밑에서 compute를 통해 반환)

    val_g_loss /= total_samples
    val_g_l2_loss /= total_samples
    val_g_adv_loss /= total_samples
    val_d_loss /= total_samples
    psnr_value = psnr.compute().item()  # 명시적으로 값을 가져옴
    ssim_value = ssim.compute().item()

    writer.add_scalar("Validation/G_Loss", val_g_loss, epoch)
    writer.add_scalar("Validation/G_L2_Loss", val_g_l2_loss, epoch)
    writer.add_scalar("Validation/G_adv_Loss", val_g_adv_loss, epoch)
    writer.add_scalar("Validation/D_Loss", val_d_loss, epoch)
    writer.add_scalar("Validation/PSNR", psnr_value, epoch)
    writer.add_scalar("Validation/SSIM", ssim_value, epoch)


    return val_g_loss, val_g_l2_loss, val_g_adv_loss, val_d_loss, psnr_value, ssim_value
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
    save_dir = "/content/drive/MyDrive/inpaint_result/CASIA_Lamp/VCNet_lr_0000100001_L1_0.8_noSEMTEXloss_beta_paper_fold2_colab/db2_train"
    writer = SummaryWriter(os.path.join(save_dir, 'SR_Stage_4%s' % datetime.now().strftime("%Y%m%d-%H%M%S")))

    train_image_paths = '/content/dataset/reflection_random(50to1.7)_db2_224_trainset'  # List of input image paths
    train_mask_paths = '/content/dataset/CASIA_Lamp/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db2_test_layer12_0.3_only_mask_trainset'  # List of mask paths
    train_gt_paths = "/content/dataset/db2_224_for_gt_inpainting_trainset"  # List of ground truth paths
    train_large_mask_paths = "/content/dataset/CASIA_Lamp/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db2_test_layer12_0.3_only_mask_h2.8_w3_trainset"  # List of ground truth paths

    val_image_paths = '/content/dataset/reflection_random(50to1.7)_db2_224_validset'  # List of input image paths
    val_mask_paths = '/content/dataset/CASIA_Lamp/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db2_test_layer12_0.3_only_mask_validset'  # List of mask paths
    val_gt_paths = "/content/dataset/db2_224_for_gt_inpainting_validset"  # List of ground truth paths
    val_large_mask_paths = "/content/dataset/CASIA_Lamp/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db2_test_layer12_0.3_only_mask_h2.8_w3_validset"  # List of ground truth paths
    
    # save_dir = r"D:\inpaint_result\CASIA_Lamp\AOT_GAN_fold1_py/db1_train"
    # writer = SummaryWriter(os.path.join(save_dir, 'SR_Stage_4%s' % datetime.now().strftime("%Y%m%d-%H%M%S")))
    #
    # train_image_paths = r"C:\Users\8138\Desktop\DB\CASIAv4\CASIA_Iris_Lamp\reflection_random(50to1.7)_db1_224_trainset"  # List of input image paths
    # train_mask_paths = r"D:\mask\CASIA_Lamp\algorithm\450to50000_174x174padding_if_gac1_4000_algorithm\db1_test_layer12_0.3_only_mask_trainset"  # List of mask paths
    # train_gt_paths = r"C:\Users\8138\Desktop\DB\CASIAv4\CASIA_Iris_Lamp\db1_224_for_gt_inpainting_trainset"  # List of ground truth paths
    # train_large_mask_paths = r"D:\mask\CASIA_Lamp\algorithm\450to50000_174x174padding_if_gac1_4000_algorithm\db1_test_layer12_0.3_only_mask_h2.8_w3_trainset"  # List of large_mask paths
    #
    # val_image_paths = r"C:\Users\8138\Desktop\DB\CASIAv4\CASIA_Iris_Lamp\reflection_random(50to1.7)_db1_224_validset"  # List of input image paths
    # val_mask_paths = r"D:\mask\CASIA_Lamp\algorithm\450to50000_174x174padding_if_gac1_4000_algorithm\db1_test_layer12_0.3_only_mask_validset"  # List of mask paths
    # val_gt_paths = r"C:\Users\8138\Desktop\DB\CASIAv4\CASIA_Iris_Lamp\db1_224_for_gt_inpainting_validset"  # List of ground truth paths
    # val_large_mask_paths = r"D:\mask\CASIA_Lamp\algorithm\450to50000_174x174padding_if_gac1_4000_algorithm\db1_test_layer12_0.3_only_mask_h2.8_w3_validset"  # List of large_mask paths

    results_path = os.path.join(save_dir, "metrics.csv")

    os.makedirs(save_dir, exist_ok=True)

    # checkpoint_path = "D:/inpaint_result/CASIA_Distance/TT-Unet_GAN_D_100x100/db1_train_2/checkpoint_epoch_106.pth.tar"  # 불러올 시 마지막 저장된 pth파일 경로 입력!!
    checkpoint_path = None

    # Parameters
    batch_size = 8
    lr_g = 0.00001
    lr_d = 0.00001
    num_epochs = 400
    lambda_adv = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # Dataset and Dataloader
    train_dataset = InpaintDataset(train_image_paths, train_mask_paths, train_gt_paths, train_large_mask_paths)
    val_dataset = InpaintDataset(val_image_paths, val_mask_paths, val_gt_paths, val_large_mask_paths)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    # Models and losses
    generator = RIN().to(device)
    mpn = MPN().to(device)

    discriminator = Discriminator().to(device)
    patchdiscriminator = PatchDiscriminator().to(device)

    IDMRFloss = IDMRFLoss().to(device)
    SemanticConsistencyloss = SemanticConsistencyLoss().to(device)
    WeightedBCEloss = WeightedBCELoss().to(device)




    # Optimizers

    optimizer_g = optim.Adam(
        list(generator.parameters()) + list(mpn.parameters()),
        lr=lr_g, betas=(0.5, 0.999)
    )

    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # 그래프 local에 저장 위한
    g_losses = []  # train loss
    g_l2_losses = []  # train loss
    g_adv_losses = []  # train loss
    g_recog_losses = []  # [추가] Feature L1 Loss 저장
    d_losses = []  # train los

    # Loss function
    criterion = nn.BCELoss()   # nn.BCELoss()는 nn.BCEWithLogitsLoss()와 달리 확률값을 입력으로 받음. discriminator가 sigmoid를 마지막에 거치므로 확률값이니 이걸 사용


    with open(results_path, mode='w', newline='') as csvfile:
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow([
            "epoch", "g_loss", "g_l2_loss", "g_adv_loss", "d_loss",
            "val_g_loss", "val_g_l2_loss", "val_g_adv_loss", "val_d_loss",
            "psnr", "ssim"
        ])


    start_epoch = 0

    if checkpoint_path:
        generator, discriminator, optimizer_g, optimizer_d, start_epoch, g_loss, g_l2_loss, d_loss \
            = load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d)


        print(f"Resuming training from epoch {start_epoch + 1}")



    for epoch in range(start_epoch, num_epochs):

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        g_loss, g_l2_loss, g_adv_loss,  d_loss = train_gan_epoch(
            generator=generator, discriminator=discriminator, patchdiscriminator=patchdiscriminator , mpn=mpn,          # model
            dataloader= train_loader,                                      # dataloader
            optimizer_g = optimizer_g,  optimizer_d = optimizer_d,          # optim
            device = device,                                               # device
            texture_consistency_loss = IDMRFloss, semantic_consistency_loss = SemanticConsistencyloss, weighted_bce_loss = WeightedBCEloss       # loss
            )
        g_losses.append(g_loss)
        g_l2_losses.append(g_l2_loss)
        g_adv_losses.append(g_adv_loss)
        d_losses.append(d_loss)

        # Validation metrics and sample image saving
        val_g_loss, val_g_l2_loss, val_g_adv_loss, val_d_loss, psnr, ssim = validate_epoch(
            generator=generator,
            mpn=mpn,
            discriminator=discriminator,
            dataloader= val_loader,
            device=device,
            writer=writer,
            epoch=epoch,

            save_dir=save_dir)


        writer.add_scalar("Train/G_Loss", g_loss, epoch)
        writer.add_scalar("Train/G_L2", g_l2_loss, epoch)
        writer.add_scalar("Train/G_adv", g_adv_loss, epoch)
        writer.add_scalar("Train/D_Loss", d_loss, epoch)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], G_Loss: {g_loss:.4f}, g_l2_Loss: {g_l2_loss:.4f}, d_Loss: {d_loss:.4f}")
        print("----------validation-----------")
        print(
            f"val_g_loss: {val_g_loss:.4f}, val_g_l2_loss: {val_g_l2_loss:.4f} ,val_g_adv_loss: {val_g_adv_loss:.4f}, val_d_loss: {val_d_loss:.4f} ,PSNR: {psnr:.4f}, SSIM: {ssim:.4f} ")

        with open(results_path, mode='a', newline='') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow([
                epoch + 1, g_loss, g_l2_loss, g_adv_loss, d_loss,
                val_g_loss, val_g_l2_loss, val_g_adv_loss, val_d_loss,
                psnr, ssim
            ])

        if epoch >= 20:
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

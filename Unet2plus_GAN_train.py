import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Unet2plus_GAN_models import UNet_2Plus, Discriminator
from dataset import InpaintDataset
from torchvision import transforms, utils
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
# from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from pytorch_fid import fid_score
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
# from convnext import convnext_small  # [추가]

def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정
seed_everything(42)

def train_gan_epoch(generator, discriminator, dataloader, criterion, optimizer_g, optimizer_d, device, lambda_adv=0.1):  # 한 에포크 학습 정의
    generator.train()
    discriminator.train()
    epoch_g_loss, epoch_g_l2_loss, epoch_g_adv_loss, epoch_d_loss = 0.0, 0.0, 0.0, 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for inputs, gts, masks, _ ,_ ,_ in progress_bar:
        inputs, gts, masks = inputs.to(device), gts.to(device), masks.to(device)

        # Train Discriminator
        optimizer_d.zero_grad()
        fake_images = generator(inputs)
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
        fake_output = discriminator(fake_images)
        g_loss_adv = criterion(fake_output, torch.ones_like(fake_output).to(device))  # Discriminator를 잘 속이는지에 대한 지표(loss)
        # g_loss_pixel = nn.MSELoss()(fake_images, gts)  # Discriminator와 관계없이 gt image와 비교했을 때 잘 복원했는지에 대한 지표(loss)

        g_loss_pixel = nn.MSELoss()(fake_images*(1-masks), gts*(1-masks)) + 50*nn.MSELoss()(fake_images*masks, gts*masks)  # Discriminator와 관계없이 gt image와 비교했을 때 잘 복원했는지에 대한 지표(loss)
        g_loss = g_loss_pixel + lambda_adv * g_loss_adv
        g_loss.backward()
        optimizer_g.step()

        epoch_g_loss += g_loss.item()
        epoch_g_l2_loss += g_loss_pixel.item()
        epoch_g_adv_loss += g_loss_adv.item()
        epoch_d_loss += d_loss.item()

        # Update progress bar with current losses
        progress_bar.set_postfix({"G_Loss": g_loss.item(), "G_L2_Loss": g_loss_pixel.item(), "G_adv_loss": g_loss_adv.item(), "D_Loss": d_loss.item()})

    return epoch_g_loss / len(dataloader), epoch_g_l2_loss / len(dataloader), epoch_g_adv_loss / len(dataloader), epoch_d_loss / len(dataloader)


def validate_epoch(generator, discriminator, dataloader, device, criterion, writer, lambda_adv, epoch,save_dir=None):
    generator.eval()
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    val_g_loss, val_g_l2_loss, val_g_adv_loss = 0.0, 0.0, 0.0
    val_d_loss= 0.0

    # Create a new directory for the epoch
    epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch + 1}')
    os.makedirs(epoch_save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (inputs, gts, masks, _, _, _) in enumerate(dataloader):
            inputs, gts, masks = inputs.to(device), gts.to(device), masks.to(device)
            fake_images = generator(inputs)

            # Save a few sample images

            if i < 5:  # Save up to 5 sample images per epoch
                # Convert from -1~1 to 0~1 for saving
                images = inputs[:, :3, :, :] # concat된 마스크 제외한 input 이미지만

                sample_images = fake_images.clamp(0, 1).cpu().numpy()  # Shape: (B, C, H, W)
                gt_images = gts.clamp(0, 1).cpu().numpy()
                input_images = images.clamp(0,1).cpu().numpy()  # Use only the first 3 channels for input

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

            g_loss_adv = criterion(discriminator(fake_images),
                                   torch.ones_like(discriminator(fake_images)).to(device))  # Discriminator를 잘 속이는지에 대한 지표(loss)
            # g_loss_pixel = nn.MSELoss()(fake_images, gts)  # Discriminator와 관계없이 gt image와 비교했을 때 잘 복원했는지에 대한 지표(loss)

            g_loss_pixel = nn.MSELoss()(fake_images*(1-masks), gts*(1-masks)) + 50*nn.MSELoss()(fake_images*masks, gts*masks)  # Discriminator와 관계없이 gt image와 비교했을 때 잘 복원했는지에 대한 지표(loss)
            g_loss = g_loss_pixel + lambda_adv * g_loss_adv

            
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


            val_g_loss += g_loss.item()
            val_g_l2_loss += g_loss_pixel.item()
            val_g_adv_loss += g_loss_adv.item()
            val_d_loss += d_loss.item()
            psnr(fake_images, gts)  # 계속 누적됨 (각 배치당 psnr이 누적(배치size로 평균처리)돼서 한 epoch를 채우면 밑에서 compute를 통해 반환)
            ssim(fake_images, gts)  # 계속 누적됨 (각 배치당 ssim이 누적(배치size로 평균처리)돼서 한 epoch를 채우면 밑에서 compute를 통해 반환)

    val_g_loss /= len(dataloader)
    val_g_l2_loss /= len(dataloader)
    val_g_adv_loss /= len(dataloader)
    val_d_loss /= len(dataloader)
    psnr_value = psnr.compute().item()  # 명시적으로 값을 가져옴
    ssim_value = ssim.compute().item()

    writer.add_scalar("Validation/G_Loss", val_g_loss, epoch)
    writer.add_scalar("Validation/G_L2_Loss", val_g_l2_loss, epoch)
    writer.add_scalar("Validation/G_adv_Loss", val_g_adv_loss, epoch)
    writer.add_scalar("Validation/D_Loss", val_d_loss, epoch)
    writer.add_scalar("Validation/PSNR", psnr_value, epoch)
    writer.add_scalar("Validation/SSIM", ssim_value, epoch)


    return val_g_loss, val_g_l2_loss, val_g_adv_loss, val_d_loss, psnr_value, ssim_value

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
    
    save_dir = "/content/drive/MyDrive/inpaint_result/CASIA_Lamp/Unet2plus_GAN_add_L2_hole_loss_50_D70x70/db1_train"
    writer = SummaryWriter(os.path.join(save_dir, 'SR_Stage_4%s' % datetime.now().strftime("%Y%m%d-%H%M%S")))

    train_image_paths = '/content/dataset//reflection_random(50to1.7)_db1_224_trainset'  # List of input image paths
    train_mask_paths = '/content/dataset/CASIA_Lamp/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db1_test_layer12_0.3_only_mask_trainset'  # List of mask paths
    train_gt_paths = "/content/dataset//db1_224_for_gt_inpainting_trainset"  # List of ground truth paths
    train_large_mask_paths = "/content/dataset/CASIA_Lamp/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db1_test_layer12_0.3_only_mask_large_mask_h2_w2.2_trainset"  # List of ground truth paths

    val_image_paths = '/content/dataset//reflection_random(50to1.7)_db1_224_validset'  # List of input image paths
    val_mask_paths = '/content/dataset/CASIA_Lamp/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db1_test_layer12_0.3_only_mask_validset'  # List of mask paths
    val_gt_paths = "/content/dataset//db1_224_for_gt_inpainting_validset"  # List of ground truth paths
    val_large_mask_paths = "/content/dataset/CASIA_Lamp/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db1_test_layer12_0.3_only_mask_large_mask_h2_w2.2_validset"  # List of ground truth paths

    results_path = os.path.join(save_dir, "metrics.csv")

    os.makedirs(save_dir, exist_ok=True)

    # checkpoint_path = "D:/inpaint_result/CASIA_Distance/TT-Unet_GAN/db1_train_2/checkpoint_epoch_106.pth.tar"  # 불러올 시 마지막 저장된 pth파일 경로 입력!!
    checkpoint_path = None

    # Parameters
    batch_size = 8
    lr = 0.0002
    num_epochs = 300
    lambda_adv = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Dataset and Dataloader
    train_dataset = InpaintDataset(train_image_paths, train_mask_paths, train_gt_paths, train_large_mask_paths)
    val_dataset = InpaintDataset(val_image_paths, val_mask_paths, val_gt_paths, val_large_mask_paths)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Models
    generator = UNet_2Plus().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # 그래프 local에 저장 위한
    g_losses = []  # train loss
    g_l2_losses = []  # train loss
    g_adv_losses = []  # train loss
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
        generator, discriminator, optimizer_g, optimizer_d, start_epoch, g_loss, g_l2_loss, g_adv_loss, d_loss \
            = load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d)


        print(f"Resuming training from epoch {start_epoch + 1}")


    for epoch in range(start_epoch, num_epochs):

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        g_loss, g_l2_loss,g_adv_loss,  d_loss = train_gan_epoch(generator, discriminator, train_loader, criterion, optimizer_g, optimizer_d, device, lambda_adv)
        g_losses.append(g_loss)
        g_l2_losses.append(g_l2_loss)
        g_adv_losses.append(g_adv_loss)
        d_losses.append(d_loss)

        # Validation metrics and sample image saving
        val_g_loss, val_g_l2_loss, val_g_adv_loss, val_d_loss, psnr, ssim = validate_epoch(generator,discriminator, val_loader, device, criterion, writer, lambda_adv, epoch, save_dir)


        writer.add_scalar("Train/G_Loss", g_loss, epoch)
        writer.add_scalar("Train/G_L2", g_l2_loss, epoch)
        writer.add_scalar("Train/G_adv", g_adv_loss, epoch)
        writer.add_scalar("Train/D_Loss", d_loss, epoch)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], G_Loss: {g_loss:.4f}, g_l2_Loss: {g_l2_loss:.4f}, g_adv_Loss: {g_adv_loss:.4f}, d_Loss: {d_loss:.4f}")
        print("----------validation-----------")
        print(
            f"val_g_loss: {val_g_loss:.4f}, val_g_l2_loss: {val_g_l2_loss:.4f}, ,val_g_adv_loss: {val_g_adv_loss:.4f}, val_d_loss: {val_d_loss:.4f} ,PSNR: {psnr:.4f}, SSIM: {ssim:.4f} ")

        with open(results_path, mode='a', newline='') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow([
                epoch + 1, g_loss, g_l2_loss, g_adv_loss, d_loss,
                val_g_loss, val_g_l2_loss, val_g_adv_loss, val_d_loss,
                psnr, ssim
            ])


        # Save checkpoint
        if epoch >= 200:
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

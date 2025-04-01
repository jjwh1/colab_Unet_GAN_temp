import torch
import os
import cv2
import pandas as pd
from torch.utils.data import DataLoader
from models import GatedGenerator
from dataset_test import InpaintDataset
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

# test_lamp.py

# Load dataset


# 수정된 test_model
def test_model(generator, dataloader, device, output_dir, results_path):
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)

    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    results = {"filename": [], "psnr": [], "ssim": []}

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing", leave=False)
        for idx, (inputs, gts, masks, filenames, large_masks) in enumerate(progress_bar):  # 이미 DataLoader가 배치 처리를 수행
            inputs = inputs.to(device)
            gts = gts.to(device)
            large_masks = large_masks.to(device)
            masks = masks.to(device)


            _,outputs = generator(inputs, masks, large_masks)
            outputs = inputs * (1 - large_masks) + outputs * large_masks
            outputs = (outputs + 1) / 2.0


            # Calculate metrics
            psnr_value = psnr_metric(outputs, gts).item()
            ssim_value = ssim_metric(outputs, gts).item()

            # Append results
            results["filename"].extend(filenames)
            results["psnr"].append(psnr_value)
            results["ssim"].append(ssim_value)

            # Save the output image, input image, and ground truth
            for i in range(outputs.size(0)):
                inpainted_image = outputs[i].cpu().numpy().transpose(1, 2, 0) * 255.0
                inpainted_image = inpainted_image.clip(0, 255).astype('uint8')

                input_image = inputs[i, :3, :, :].cpu().numpy().transpose(1, 2, 0) * 255.0
                input_image = input_image.clip(0, 255).astype('uint8')

                gt_image = gts[i].cpu().numpy().transpose(1, 2, 0) * 255.0
                gt_image = gt_image.clip(0, 255).astype('uint8')

                # Save images using the original filename
                filename = filenames[i]
                cv2.imwrite(os.path.join(output_dir, f"{filename}"), cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR))


    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    # Paths
    test_image_dir = r"C:\Users\8138\Desktop\DB\CASIAv4\CASIA_Iris_Lamp\reflection_random(50to1.7)_db1_224"  # Replace with your test image directory
    test_mask_dir = 'D:/mask/CASIA_Lamp/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db1_test_layer12_0.3_only_mask'  # Replace with your test mask directory
    test_gt_dir = r"C:\Users\8138\Desktop\DB\CASIAv4\CASIA_Iris_Lamp\db1_224_for_gt_inpainting"        # Replace with your test ground truth directory

    output_dir = r"D:\inpaint_result\CASIA_Lamp\TT_GAN_fold2_py_\db2_train\db1_test_epoch_365_copy_h2.8_w3"          # Replace with your desired output directory
    results_path = os.path.join(output_dir, "test_results.csv")
    checkpoint_path = r"D:\inpaint_result\CASIA_Lamp\TT_GAN_fold2_py_\db2_train\checkpoint_epoch_365.pth.tar"

    large_mask_dir = "D:/mask/CASIA_Lamp/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db1_test_layer12_0.3_only_mask_h2.8_w3"


    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    test_dataset = InpaintDataset(test_image_dir, test_mask_dir, test_gt_dir, large_mask_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # collate_fn 제거

    # Load model
    generator = GatedGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])

    # Run test
    test_model(generator, test_loader, device, output_dir, results_path)









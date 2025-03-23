import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
import numpy as np
from AUIFNet_2_4 import Encoder_Base, Encoder_Detail, Decoder, AdaptiveWaveletFusion, apply_tv_denoising, dwt_decompose, apply_lbp
from args_2_4 import train_path, train_data_path, device, batch_size, img_size, is_cuda
import kornia
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import scipy.stats as stats
import cv2

test_data_path = train_data_path

transform_pipeline = T.Compose([
    T.RandomResizedCrop(img_size),
    T.Grayscale(1),
    T.ToTensor(),
])

test_dataset = torchvision.datasets.ImageFolder(test_data_path, transform=transform_pipeline)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

Encoder_Base_Test = Encoder_Base().to(device)
Encoder_Detail_Test = Encoder_Detail().to(device)
Decoder_Test = Decoder().to(device)
Adaptive_Fusion_Test = AdaptiveWaveletFusion().to(device)

Encoder_Base_Test.load_state_dict(torch.load(os.path.join(train_path, "Encoder_Base_trained.model"), weights_only=True))
Encoder_Detail_Test.load_state_dict(torch.load(os.path.join(train_path, "Encoder_Detail_trained.model"), weights_only=True))
Decoder_Test.load_state_dict(torch.load(os.path.join(train_path, "Decoder_trained.model"), weights_only=True))
Adaptive_Fusion_Test.load_state_dict(torch.load(os.path.join(train_path, "AdaptiveWaveletFusion_trained.model"), weights_only=True))

Encoder_Base_Test.eval()
Encoder_Detail_Test.eval()
Decoder_Test.eval()
Adaptive_Fusion_Test.eval()

def compute_psnr(img1, img2):
    img1_np = img1.cpu().numpy().squeeze()
    img2_np = img2.cpu().numpy().squeeze()
    return psnr(img1_np, img2_np, data_range=1.0)

def compute_ssim(img1, img2):
    img1_np = img1.cpu().numpy().squeeze()
    img2_np = img2.cpu().numpy().squeeze()
    return ssim(img1_np, img2_np, data_range=1.0)

def compute_mi(img1, img2):
    img1_np = (img1 * 255).cpu().numpy().squeeze().astype(np.uint8)
    img2_np = (img2 * 255).cpu().numpy().squeeze().astype(np.uint8)
    hist_2d, _, _ = np.histogram2d(img1_np.ravel(), img2_np.ravel(), bins=256)
    return stats.entropy(hist_2d.sum(axis=0)) + stats.entropy(hist_2d.sum(axis=1)) - stats.entropy(hist_2d.ravel())

def compute_entropy(img):
    img_np = img.cpu().numpy().squeeze()
    img_np = np.clip(img_np, 0, 1)  # Ensure 0-1 range
    hist, _ = np.histogram(img_np, bins=256, range=(0, 1), density=True)
    hist = hist + 1e-10  # Avoid log(0)
    return -np.sum(hist * np.log2(hist))

def compute_sf(img):
    img_np = img.cpu().numpy().squeeze()
    dx = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(np.mean(dx**2) + np.mean(dy**2))

def compute_sd(img):
    img_np = img.cpu().numpy().squeeze()
    return np.std(img_np)

def main():
    output_dir = os.path.join(train_path, "test_outputs")
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        'PSNR_IR': [], 'PSNR_VIS': [], 'SSIM_IR': [], 'SSIM_VIS': [],
        'MI': [], 'Entropy': [], 'SF': [], 'SD': []
    }

    print("============ Testing Begins ===============")
    with torch.no_grad():
        for i, (img_input, _) in enumerate(test_loader):
            img_input = img_input.to(device)
            half_batch = img_input.size(0) // 2
            img_ir = img_input[:half_batch]
            img_vis = img_input[half_batch:half_batch*2]

            B_K_ir, _, _ = Encoder_Base_Test(img_ir)
            D_K_ir, _, _ = Encoder_Detail_Test(img_ir)
            B_K_vis, _, _ = Encoder_Base_Test(img_vis)
            D_K_vis, _, _ = Encoder_Detail_Test(img_vis)

            wavelet_ir = torch.stack([B_K_ir, D_K_ir, D_K_ir, D_K_ir], dim=1)
            wavelet_vis = torch.stack([B_K_vis, D_K_vis, D_K_vis, D_K_vis], dim=1)

            fused_wavelet = Adaptive_Fusion_Test(wavelet_ir, wavelet_vis)
            F_b, F_d = fused_wavelet[:, 0], fused_wavelet[:, 1].sum(dim=1, keepdim=True)
            img_fused = Decoder_Test(F_b, F_d)

            img_fused = F.interpolate(img_fused, size=(img_size, img_size), mode='bilinear', align_corners=False)
            img_fused = apply_lbp(img_fused)
            img_fused = torch.clamp(img_fused, 0, 1)  # Ensure 0-1 range

            for j in range(half_batch):
                ir = img_ir[j:j+1]
                vis = img_vis[j:j+1]
                fused = img_fused[j:j+1]

                metrics['PSNR_IR'].append(compute_psnr(ir, fused))
                metrics['PSNR_VIS'].append(compute_psnr(vis, fused))
                metrics['SSIM_IR'].append(compute_ssim(ir, fused))
                metrics['SSIM_VIS'].append(compute_ssim(vis, fused))
                metrics['MI'].append(compute_mi(ir, fused))
                metrics['Entropy'].append(compute_entropy(fused))
                metrics['SF'].append(compute_sf(fused))
                metrics['SD'].append(compute_sd(fused))

                fused_img = fused.squeeze().cpu().numpy() * 255
                fused_img = fused_img.astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f"F_{i*half_batch + j + 1}.png"), fused_img)

    print("\nAverage Performance Metrics:")
    for metric, values in metrics.items():
        avg = np.mean(values)
        print(f"{metric}: {avg:.6f}")

    print("Testing completed.")

if __name__ == '__main__':
    main()
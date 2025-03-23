import torchvision
from torchvision import transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from efficientnet_pytorch import EfficientNet
from skimage.restoration import denoise_tv_chambolle
from skimage.feature import local_binary_pattern
from args_2_4 import train_data_path, train_path, device, batch_size, lr, is_cuda, log_interval, img_size, epochs

# TV Denoising Preprocessing
def apply_tv_denoising(img):
    img_np = img.cpu().detach().numpy()
    denoised = np.zeros_like(img_np)
    for b in range(img_np.shape[0]):
        for c in range(img_np.shape[1]):
            denoised[b, c] = denoise_tv_chambolle(img_np[b, c], weight=0.1, channel_axis=None)
    return torch.from_numpy(denoised).to(device)

# Multi-level DWT Decomposition
def dwt_decompose(img, levels=2, wavelet='haar'):
    batch_size, channels, height, width = img.shape
    coeffs_list = []
    for b in range(batch_size):
        batch_coeffs = []
        for c in range(channels):
            img_np = img[b, c].cpu().detach().numpy()
            coeffs = pywt.wavedec2(img_np, wavelet, level=levels)
            batch_coeffs.append(coeffs[0])  # Only LL band
        coeffs_list.append(torch.from_numpy(np.stack(batch_coeffs)))
    LL = torch.stack(coeffs_list).to(device)  # [batch_size, channels, h//4, w//4]
    return LL, None  # Simplified: only LL for now

# LBP Post-processing (fixed for integers)
def apply_lbp(img, radius=3, n_points=8):
    img_np = (img * 255).cpu().detach().numpy().astype(np.uint8)  # Scale to 0-255, integer
    lbp = np.zeros_like(img_np, dtype=np.float32)  # Float output for PyTorch
    for b in range(img_np.shape[0]):
        for c in range(img_np.shape[1]):
            lbp[b, c] = local_binary_pattern(img_np[b, c], n_points, radius, method='uniform')
    return torch.from_numpy(lbp / 255.0).to(device)  # Back to 0-1 range

class Encoder_Base(nn.Module):
    def __init__(self, size=img_size, wavelet='haar'):
        super(Encoder_Base, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.wavelet = wavelet
        self.size = size
        self.proj = nn.Conv2d(1280, 64, kernel_size=1)
        self.final_proj = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, img):
        img = apply_tv_denoising(img)
        batch_size, channels, height, width = img.shape
        features = self.efficientnet.extract_features(img)
        features = self.proj(features)
        features = F.interpolate(features, size=(height, width), mode='bilinear', align_corners=False)
        LL, _ = dwt_decompose(features, wavelet=self.wavelet)
        return self.final_proj(LL), [], []

class Encoder_Detail(nn.Module):
    def __init__(self, size=img_size, wavelet='haar'):
        super(Encoder_Detail, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.wavelet = wavelet
        self.size = size
        self.proj = nn.Conv2d(1280, 64, kernel_size=1)
        self.final_proj = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, img):
        img = apply_tv_denoising(img)
        batch_size, channels, height, width = img.shape
        features = self.efficientnet.extract_features(img)
        features = self.proj(features)
        features = F.interpolate(features, size=(height, width), mode='bilinear', align_corners=False)
        LL, _ = dwt_decompose(features, wavelet=self.wavelet)
        return self.final_proj(LL), [], []

class Decoder(nn.Module):
    def __init__(self, wavelet='haar'):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 128, 3, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.wavelet = wavelet
    
    def forward(self, fm_b, fm_d):
        fm = fm_b + fm_d
        return self.decoder(fm)

class AdaptiveWaveletFusion(nn.Module):
    def __init__(self, input_channels=1):
        super(AdaptiveWaveletFusion, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)
        self.softmax = nn.Softmax(dim=1)
        self.ir_weight = 2.5
        self.vis_weight = 1.5
    
    def forward(self, wavelet_ir, wavelet_vis):
        ir_mean = wavelet_ir.mean(dim=[2, 3, 4])
        vis_mean = wavelet_vis.mean(dim=[2, 3, 4])
        x = torch.cat([ir_mean, vis_mean], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        fusion_weights = torch.clamp(self.softmax(self.fc3(x)), 0.35, 0.55)
        fusion_weights = fusion_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        fused_wavelet = torch.zeros_like(wavelet_ir)
        fused_wavelet[:, 0] = fusion_weights[:, 0] * wavelet_ir[:, 0] + (1 - fusion_weights[:, 0]) * wavelet_vis[:, 0]
        fused_wavelet[:, 1:] = (fusion_weights[:, 1:] * self.ir_weight * wavelet_ir[:, 1:] + 
                                (1 - fusion_weights[:, 1:]) * self.vis_weight * wavelet_vis[:, 1:]) / (self.ir_weight + self.vis_weight)
        return fused_wavelet

def train():
    transform_pipeline = T.Compose([
        T.RandomResizedCrop(img_size),
        T.Grayscale(1),
        T.ToTensor(),
    ])
    Data = torchvision.datasets.ImageFolder(train_data_path, transform=transform_pipeline)
    dataloader = torch.utils.data.DataLoader(Data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    Encoder_Base_Train = Encoder_Base(wavelet='haar').to(device)
    Encoder_Detail_Train = Encoder_Detail(wavelet='haar').to(device)
    Decoder_Train = Decoder(wavelet='haar').to(device)
    Adaptive_Fusion_Train = AdaptiveWaveletFusion().to(device)

    optimizer1 = optim.Adam(Encoder_Base_Train.parameters(), lr=8e-4)
    optimizer2 = optim.Adam(Encoder_Detail_Train.parameters(), lr=8e-4)
    optimizer3 = optim.Adam(Decoder_Train.parameters(), lr=8e-4)
    optimizer4 = optim.Adam(Adaptive_Fusion_Train.parameters(), lr=8e-4)

    scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, [6, 10], gamma=0.1)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, [6, 10], gamma=0.1)
    scheduler3 = optim.lr_scheduler.MultiStepLR(optimizer3, [6, 10], gamma=0.1)
    scheduler4 = optim.lr_scheduler.MultiStepLR(optimizer4, [6, 10], gamma=0.1)

    MSELoss = nn.MSELoss()
    SSIMLoss = kornia.losses.SSIMLoss(3, reduction='mean')

    print('============ Training Begins ===============')
    total_images_processed = 0
    for epoch in range(16):
        Encoder_Base_Train.train()
        Encoder_Detail_Train.train()
        Decoder_Train.train()
        Adaptive_Fusion_Train.train()
        
        for step, (img_input, _) in enumerate(dataloader):
            batch_size_current = img_input.size(0)
            total_images_processed += batch_size_current
            
            img_input = img_input.to(device)
            img_ir = img_input[:batch_size_current//2]
            img_vis = img_input[batch_size_current//2:]
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            
            B_K_ir, _, _ = Encoder_Base_Train(img_ir)
            D_K_ir, _, _ = Encoder_Detail_Train(img_ir)
            B_K_vis, _, _ = Encoder_Base_Train(img_vis)
            D_K_vis, _, _ = Encoder_Detail_Train(img_vis)
            
            wavelet_ir = torch.stack([B_K_ir, D_K_ir, D_K_ir, D_K_ir], dim=1)
            wavelet_vis = torch.stack([B_K_vis, D_K_vis, D_K_vis, D_K_vis], dim=1)
            
            fused_wavelet = Adaptive_Fusion_Train(wavelet_ir, wavelet_vis)
            F_b, F_d = fused_wavelet[:, 0], fused_wavelet[:, 1].sum(dim=1, keepdim=True)
            img_fused = Decoder_Train(F_b, F_d)
            
            # Move LBP out of gradient path (apply post-training or in test)
            img_fused_for_loss = F.interpolate(img_fused, size=(img_size, img_size), mode='bilinear', align_corners=False)
            
            mse_loss = MSELoss(img_ir, img_fused_for_loss)
            ssim_loss = SSIMLoss(img_ir, img_fused_for_loss)
            entropy_loss = -torch.mean(img_fused_for_loss * torch.log(img_fused_for_loss + 1e-10))
            loss = 2.0 * mse_loss + 3.5 * ssim_loss + 0.1 * entropy_loss
            
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            
            # Apply LBP only for logging/output, not gradient
            with torch.no_grad():
                img_fused = apply_lbp(img_fused)
            
            print(f'Epoch {epoch + 1}/16, Step {step + 1}, Loss: {loss.item():.6f}, Images Processed: {total_images_processed}')
        
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        scheduler4.step()

    for name, model in zip(['Encoder_Base', 'Encoder_Detail', 'Decoder', 'AdaptiveWaveletFusion'], 
                           [Encoder_Base_Train, Encoder_Detail_Train, Decoder_Train, Adaptive_Fusion_Train]):
        save_path = f"{train_path}/{name}_trained.model"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

if __name__ == '__main__':
    train()
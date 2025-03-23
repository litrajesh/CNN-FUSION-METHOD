import torchvision
from torchvision import transforms
import torch
from torch import amp
import kornia
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from AUIFNet_2_4 import Encoder_Base, Encoder_Detail, Decoder, AdaptiveWaveletFusion, apply_tv_denoising, dwt_decompose, apply_lbp
from args_2_4 import train_data_path, train_path, device, batch_size, lr, is_cuda, log_interval, img_size, epochs

transforms = transforms.Compose([
    transforms.RandomResizedCrop(img_size),
    transforms.Grayscale(1),
    transforms.ToTensor(),
])

Data = torchvision.datasets.ImageFolder(train_data_path, transform=transforms)
dataloader = torch.utils.data.DataLoader(Data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)  # Drop last uneven batch

Encoder_Base_Train = Encoder_Base().to(device)
Encoder_Detail_Train = Encoder_Detail().to(device)
Decoder_Train = Decoder().to(device)
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
scaler = amp.GradScaler('cuda')

def main():
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
            
            # Ensure even split for IR and VIS
            half_batch = batch_size_current // 2
            img_ir = img_input[:half_batch]
            img_vis = img_input[half_batch:half_batch*2]  # Explicitly take equal halves
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            
            with amp.autocast('cuda'):
                B_K_ir, _, _ = Encoder_Base_Train(img_ir)
                D_K_ir, _, _ = Encoder_Detail_Train(img_ir)
                B_K_vis, _, _ = Encoder_Base_Train(img_vis)
                D_K_vis, _, _ = Encoder_Detail_Train(img_vis)
                
                wavelet_ir = torch.stack([B_K_ir, D_K_ir, D_K_ir, D_K_ir], dim=1)
                wavelet_vis = torch.stack([B_K_vis, D_K_vis, D_K_vis, D_K_vis], dim=1)
                
                fused_wavelet = Adaptive_Fusion_Train(wavelet_ir, wavelet_vis)
                F_b, F_d = fused_wavelet[:, 0], fused_wavelet[:, 1].sum(dim=1, keepdim=True)
                img_fused = Decoder_Train(F_b, F_d)
                
                img_fused_for_loss = F.interpolate(img_fused, size=(img_size, img_size), mode='bilinear', align_corners=False)
                
                mse_loss = MSELoss(img_ir, img_fused_for_loss)
                ssim_loss = SSIMLoss(img_ir, img_fused_for_loss)
                entropy_loss = -torch.mean(img_fused_for_loss * torch.log(img_fused_for_loss + 1e-10))
                loss = 2.0 * mse_loss + 3.5 * ssim_loss + 0.1 * entropy_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer1)
            scaler.step(optimizer2)
            scaler.step(optimizer3)
            scaler.step(optimizer4)
            scaler.update()
            
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
    main()
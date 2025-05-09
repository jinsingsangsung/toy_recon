import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class HybridCompression(nn.Module):
    def __init__(self, height, width, rank=5, fourier_k=3, channels=3):
        super().__init__()
        # Low-rank component per channel
        U = torch.empty(channels, height, rank)
        V = torch.empty(channels, rank, width)
        nn.init.kaiming_normal_(U, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(V, mode='fan_out', nonlinearity='linear')
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)
        
        # Few Fourier components for fine details per channel
        frequencies = torch.empty(channels, fourier_k, 2)
        amplitudes = torch.empty(channels, fourier_k)
        phases = torch.empty(channels, fourier_k)
        nn.init.kaiming_normal_(frequencies, mode='fan_out', nonlinearity='linear') 
        nn.init.kaiming_normal_(amplitudes, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(phases, mode='fan_out', nonlinearity='linear')
        self.frequencies = nn.Parameter(frequencies)
        self.amplitudes = nn.Parameter(amplitudes)
        self.phases = nn.Parameter(phases)
        
        # Create coordinate grid once
        y = torch.linspace(-1, 1, height)
        x = torch.linspace(-1, 1, width)
        self.register_buffer('grid_y', y.view(-1, 1).repeat(1, width))
        self.register_buffer('grid_x', x.view(1, -1).repeat(height, 1))
    
    def forward(self):
        # Low-rank base for each channel
        result = torch.zeros(self.U.shape[0], self.U.shape[1], self.V.shape[2], device=self.U.device)
        for c in range(self.U.shape[0]):
            result[c] = self.U[c] @ self.V[c]
        
        # Add Fourier details for each channel
        for c in range(self.frequencies.shape[0]):
            for i in range(self.frequencies.shape[1]):
                freq_y, freq_x = self.frequencies[c,i]
                phase = self.phases[c,i]
                amp = self.amplitudes[c,i]
                basis = amp * torch.sin(
                    2 * np.pi * (freq_x * self.grid_x + freq_y * self.grid_y) + phase
                )
                result[c] += basis
            
        return result
    
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)

    
    # Load and resize image to 32x32
    # img = Image.open('sample_image.png')
    img = Image.open('../test_img/kodim15.png')
    # img = Image.open('./yubit.JPG')
    w, h = img.size
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor()
    ])
    model = HybridCompression(height=h, width=w, rank=32, fourier_k=3).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    x = transform(img).unsqueeze(0).cuda()
    b, c, h, w = x.shape
    
    epochs = 1000
    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        compressed = model()  # Compress
        # reconstructed = conv.reconstruct(compressed)  # Reconstruct
        # reconstructed = conv.reconstruct_2(compressed)  # Reconstruct
        
        # Calculate loss between original and reconstructed
        loss = criterion(compressed[None], x)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Step the scheduler
        scheduler.step()
        
        if (epoch + 1) % 100 == 0:
            current_lrs = scheduler.get_last_lr()  # Get all learning rates
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, LR: {current_lrs[0]:.6f}')

    with torch.no_grad():
        import time
        start_time = time.time()
        # final_compressed = model()
        final_reconstructed = model()
        # final_reconstructed = conv.reconstruct_2(final_compressed)
        elapsed_time = time.time() - start_time
        print(f"Inference time: {elapsed_time:.4f} seconds")
        final_loss = criterion(final_reconstructed[None], x)
        # compression_ratio_2 = x.numel() / (final_compressed.numel() + conv.P*kernel_size + conv.P*conv.H)
        # Count total parameters in the model
        total_params = sum(p.numel() for p in model.parameters())
        
        # Calculate compression ratio comparing input size to model parameters
        compression_ratio = x.numel() / total_params
        print(f"Input size: {x.numel()} elements")
        print(f"Model parameters: {total_params} elements") 
        print(f"Compression ratio: {compression_ratio:.2f}x")
        # compression_ratio_naive = x.numel() / (final_compressed.numel() + conv.in_channels**2 + conv.in_channels*conv.out_channels*conv.k)
        # compression_ratio = x.numel() / (final_compressed.numel() + 3*conv.P + 4*conv.P**2 + conv.H*conv.H)
        # compression_ratio_naive = x.numel() / (final_compressed.numel() + conv.conv.in_channels*conv.conv.out_channels*conv.k1*conv.k2 + 1)
        # print(f"\nFinal reconstruction loss: {final_loss.item():.6f}")
        # print(f"Compression ratio (construct_2): {compression_ratio_2:.2f}x")
        # print(f"Compression ratio (construct): {compression_ratio:.2f}x")
        # print(f"Compression ratio (naive): {compression_ratio_naive:.2f}x")
        # print(f"embedding size:{final_compressed.shape}")
        # print(f"decoder size:{conv.P**2 + conv.P*conv.H}")
        # print(f"real compress ratio:{x.numel() / }")
        # draw_scatter_plot_conv1d() 
        # Convert reconstructed tensor back to image and save
        # Process reconstructed image
        recon_img = final_reconstructed.squeeze(0)  # Remove batch dimension
        recon_img = recon_img.view(3, h, w)  # Reshape back to image dimensions
        recon_img = torch.clamp(recon_img, 0, 1)  # Clamp values between 0 and 1
        
        # Process original image
        orig_img = x.squeeze(0)  # Remove batch dimension
        orig_img = orig_img.view(c, h, w)  # Reshape back to image dimensions
        
        # Convert both to PIL images
        transform = transforms.ToPILImage()
        recon_img = transform(recon_img)
        orig_img = transform(orig_img)
        
        # Create a new image combining both side by side
        combined_img = Image.new('RGB', (2*w, h))  # Width is doubled to fit both images
        combined_img.paste(orig_img, (0, 0))  # Original on left
        combined_img.paste(recon_img, (w, 0))  # Reconstructed on right
        
        # Save combined image
        combined_img.save('comparison.png')
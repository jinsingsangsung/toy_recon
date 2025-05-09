import torch
import torchdiffeq
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model_all import SSMConv2d
import numpy as np


class ODEUpsample(nn.Module):
    """ODE function that progressively upsamples features"""
    def __init__(self, hidden_channels, target_size, num_scales=3):
        super().__init__()
        self.target_size = target_size  # (H, W) of original input
        self.num_scales = num_scales
        
        # Progressive upsampling blocks
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
                nn.GroupNorm(min(32, hidden_channels), hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
                nn.GroupNorm(min(32, hidden_channels), hidden_channels),
                nn.SiLU(),
            ) for _ in range(num_scales)
        ])
        
        # Final processing
        self.final_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        
        # Time embedding
        time_dim = hidden_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Time projections for each scale
        self.time_projections = nn.ModuleList([
            nn.Linear(time_dim, hidden_channels)
            for _ in range(num_scales)
        ])
    
    def forward(self, t, x):
        # Time embedding
        t_emb = self.time_mlp(t.view(1, 1))  # [1, time_dim]
        
        # Progressive upsampling
        dx = x
        for i, block in enumerate(self.upsample_blocks):
            # Apply upsampling block
            dx = block(dx)
            
            # Add time-dependent modulation
            t_scale = self.time_projections[i](t_emb)
            dx = dx * (1 + 0.1 * t_scale.view(1, -1, 1, 1))
        
        # Final processing
        dx = self.final_conv(dx)
        
        # Scale for stability
        return dx * 0.1

class SSMODEDecoder(nn.Module):
    """ODE-based decoder that learns the reverse path of SSMConv2d"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding):
        super().__init__()
        
        # Forward path (known, fixed)
        self.ssm_conv = SSMConv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            blocks=8,
            C_init_method="trunc_standard_normal",
            discretization="zoh",
        )
        
        # Freeze the encoder parameters
        for param in self.ssm_conv.parameters():
            param.requires_grad = False
        
        # Calculate required number of upsampling steps
        try:
            k1, k2 = kernel_size
        except:
            k1 = k2 = kernel_size
        self.num_scales = int(np.log2(max(k1, k2)))
        
        # ODE function with upsampling
        self.ode_func = ODEUpsample(
            hidden_channels=hidden_channels,
            target_size=(k1, k2),
            num_scales=self.num_scales
        )
        
        # Final projection to input channels
        self.to_rgb = nn.Conv2d(hidden_channels, in_channels, 1)
        
        # Integration parameters
        self.integration_time = torch.linspace(0, 1, 10)  # More timesteps for better gradients
        
    def forward_dynamics(self, t, x):
        """Forward dynamics with upsampling"""
        return self.ode_func(t, x)
        
    def encode(self, x):
        """Known forward path using SSMConv2d"""
        with torch.no_grad():
            return self.ssm_conv(x)
            
    def decode(self, z):
        """Learned reverse path using ODE with upsampling"""
        trajectory = torchdiffeq.odeint(
            self.forward_dynamics,
            z,
            self.integration_time,
            method='dopri5',
            rtol=1e-4,
            atol=1e-4,
        )
        final_features = trajectory[-1]
        return self.to_rgb(final_features)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

def train_ode_decoder():
    # Initialize model
    model = SSMODEDecoder(
        in_channels=3,
        hidden_channels=128,
        kernel_size=(128, 128),
        stride=(128, 128),
        padding=(0, 0)
    ).cuda()
    
    optimizer = torch.optim.AdamW([
        {'params': model.ode_func.parameters(), 'lr': 0.001},
        {'params': model.to_rgb.parameters(), 'lr': 0.001}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    
    def compute_loss(x, x_recon):
        # Reconstruction loss
        loss_recon = mse_loss(x_recon, x)
        return loss_recon
    
    # Load data
    img = Image.open('../test_img/kodim15.png')
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(img).unsqueeze(0).cuda()
    
    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        x_recon = model(x)
        
        # Compute loss
        loss = compute_loss(x, x_recon)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
            
            # Visualize results
            if epoch % 500 == 0:
                with torch.no_grad():
                    # Compare with SSMConv2d's built-in reconstruct
                    z = model.encode(x)
                    x_recon_ode = model.decode(z)
                    x_recon_ssm = model.ssm_conv.reconstruct(z)
                    
                    # Save comparison
                    imgs = [x.cpu().squeeze(0), 
                           x_recon_ode.cpu().squeeze(0),
                           x_recon_ssm.cpu().squeeze(0)]
                    
                    # Create side-by-side comparison
                    transform = transforms.ToPILImage()
                    combined_img = Image.new('RGB', (3*img.width, img.height))
                    for i, img_tensor in enumerate(imgs):
                        img_pil = transform(torch.clamp(img_tensor, 0, 1))
                        combined_img.paste(img_pil, (i*img.width, 0))
                    combined_img.save(f'comparison_epoch_{epoch}.png')

if __name__ == "__main__":
    train_ode_decoder()
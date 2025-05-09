import torch
import torchdiffeq
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model_all import SSMConv2d
import numpy as np

class ODEFunc(nn.Module):
    """General ODE function for feature evolution"""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1)
        )
        
        # Initialize weights for stable dynamics
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, t, x):
        """
        t: scalar time point
        x: feature tensor [B, C, H, W]
        """
        # Normalize dynamics for stability
        dx = self.net(x)
        return dx * 0.1  # Scale factor to prevent explosive dynamics

class SSMODEAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding):
        super().__init__()
        
        # SSM for downscaling/upscaling
        self.ssm_conv = SSMConv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            blocks=8,
            C_init_method="trunc_standard_normal",
            discretization="zoh",
            dt_min=0.001,
            dt_max=0.1,
            conj_sym=False,
            clip_eigs=False,
            bidirectional=False,
            step_rescale=1.0,
        )
        
        # ODE function for feature evolution
        self.ode_func = ODEFunc(hidden_channels)
        
        # Integration parameters
        self.integration_time = torch.linspace(0, 1, 2)
        self.solver = 'dopri5'
        self.atol = 1e-4
        self.rtol = 1e-4
    
    def encode(self, x):
        # Downscale using SSM
        z = self.ssm_conv(x)
        
        # Evolve features through ODE
        trajectory = torchdiffeq.odeint(
            self.ode_func,
            z,
            self.integration_time,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol
        )
        return trajectory[-1]
    
    def decode(self, z):
        # Reverse ODE evolution
        backward_time = torch.flip(self.integration_time, [0])
        trajectory = torchdiffeq.odeint(
            self.ode_func,
            z,
            backward_time,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol
        )
        z_reversed = trajectory[-1]
        
        # Upscale using SSM reconstruct
        return self.ssm_conv.reconstruct(z_reversed)
    
    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z)

def train_ode_autoencoder():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)

    # Model initialization
    model = SSMODEAutoencoder(
        in_channels=3,
        hidden_channels=128,
        kernel_size=(128, 128),
        stride=(64, 64),
        padding=(32, 32),
        # kernel_size=(512, 768),
        # stride=(512, 768),
        # padding=(0, 0)
    ).cuda()
    
    ssm_params = []
    non_ssm_params = list(model.ode_func.parameters())
    for name, param in model.ssm_conv.named_parameters():
        if any(x in name for x in ['Lambda_re', 'Lambda_im', 'B_', 'log_step']):
            ssm_params.append(param)
        else:
            non_ssm_params.append(param)

    # Optimizer with different learning rates for SSM and ODE
    # import lion_pytorch
    optimizer = torch.optim.AdamW([
        {'params': ssm_params, 'lr': 0.002},
        {'params': non_ssm_params, 'lr': 0.02}
    ])
    # Print total number of parameters in millions
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params/1e6:.2f}M')
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    criterion = nn.MSELoss()
    
    # Load data
    img = Image.open('../test_img/kodim15.png')
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(img).unsqueeze(0).cuda()
    
    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        compressed, reconstructed = model(x)
        
        # Loss computation
        loss = criterion(reconstructed, x)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
            
            # Visualization
        if epoch % 500 == 0 or epoch + 1 == epochs:
            with torch.no_grad():
                # Save intermediate results
                recon_img = reconstructed.squeeze(0).cpu()
                orig_img = x.squeeze(0).cpu()
                
                transform = transforms.ToPILImage()
                recon_img = transform(torch.clamp(recon_img, 0, 1))
                orig_img = transform(orig_img)
                
                # Create side-by-side comparison
                combined_img = Image.new('RGB', (2*orig_img.width, orig_img.height))
                combined_img.paste(orig_img, (0, 0))
                combined_img.paste(recon_img, (orig_img.width, 0))
                combined_img.save(f'comparison_epoch_{epoch}.png')
                print(f'Saved comparison image for epoch {epoch}')

if __name__ == "__main__":
    train_ode_autoencoder()
# copied from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py

import numpy as np
import torch
import torch.nn as nn
# import geotorch
# import cv2
import torchvision
from lion_pytorch import Lion

try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))
    
    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
except ImportError:
    # PyTorch 1.6.0 and older versions
    def dct1_rfft_impl(x):
        return torch.rfft(x, 1)
    
    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)

    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)



def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    x = torch.cat([x, x.flip([1])[:, 1:-1]], dim=1)

    return dct1_rfft_impl(x)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, drop=0, ortho_constraint=False, ortho_freq=10, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.drop = drop
        self.norm = norm
        if drop == 0:
            self.drop_len = 0
        elif drop > 1:
            self.drop_len = drop
        else:
            self.drop_len = int(drop * in_features)

        if type[0] == 'i':
            super(LinearDCT, self).__init__(in_features-self.drop_len, in_features, bias=bias)
        else:
            super(LinearDCT, self).__init__(in_features, in_features-self.drop_len, bias=bias)
        
        self.ortho_constraint = ortho_constraint
        if ortho_constraint:
            self.steps = 0
            self.ortho_freq = ortho_freq
        

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()[:self.N-self.drop_len, :]
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()[:, :self.N-self.drop_len]
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()[:self.N-self.drop_len, :]
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()[:, :self.N-self.drop_len]
        self.weight.requires_grad = False # don't learn this!

    def forward(self, x):
        if self.ortho_constraint:
            # During training, ensure weights stay orthogonal
            if self.training:
                self.steps += 1
                if self.steps % self.ortho_freq == 0:   
                    with torch.no_grad():
                        # Perform SVD
                        U, S, V = torch.svd(self.weight)
                        # Reconstruct with identity singular values
                        self.weight.copy_(torch.mm(U, V.t()))
        return super().forward(x)        


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)

def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)

def OutImg(x, out_bias='tanh'):
    if out_bias == 'sigmoid':
        return torch.sigmoid(x)
    elif out_bias == 'tanh':
        return (torch.tanh(x) * 0.5) + 0.5
    else:
        return x + float(out_bias)



class SimpleNet(nn.Module):
    def __init__(self, W, H, drop=0.75, ortho_constraint=False, ortho_freq=1):
        super(SimpleNet, self).__init__()
        # First create the layers without constraints
        self.linear_layer_x = LinearDCT(W, 'dct', drop=drop, ortho_constraint=ortho_constraint, ortho_freq=ortho_freq)
        self.linear_layer_y = LinearDCT(H, 'dct', drop=drop, ortho_constraint=ortho_constraint, ortho_freq=ortho_freq)
        self.linear_layer_i_y = LinearDCT(H, 'idct', drop=drop, ortho_constraint=ortho_constraint, ortho_freq=ortho_freq)
        self.linear_layer_i_x = LinearDCT(W, 'idct', drop=drop, ortho_constraint=ortho_constraint, ortho_freq=ortho_freq)
        
    def forward(self, x):
        X = self.linear_layer_x(x)
        X = self.linear_layer_y(X.transpose(-1, -2))
        # print(f"shape after second transformation: {X.shape}")
        # import pdb; pdb.set_trace()
        X = self.linear_layer_i_y(X)
        recon_X = self.linear_layer_i_x(X.transpose(-1, -2))
        return recon_X

class SimpleNetEncoder(nn.Module):
    def __init__(self, W, H, drop=0.75, ortho_constraint=False, ortho_freq=1):
        super(SimpleNetEncoder, self).__init__()
        self.linear_layer_x = LinearDCT(W, 'dct', drop=drop, ortho_constraint=ortho_constraint, ortho_freq=ortho_freq)
        self.linear_layer_y = LinearDCT(H, 'dct', drop=drop, ortho_constraint=ortho_constraint, ortho_freq=ortho_freq)

    def forward(self, x):
        X = self.linear_layer_x(x)
        X = self.linear_layer_y(X.transpose(-1, -2))
        return X.transpose(-1, -2)

class SimpleNetDecoder(nn.Module):
    def __init__(self, W, H, drop=0.75, ortho_constraint=False, ortho_freq=1):
        super(SimpleNetDecoder, self).__init__()
        self.linear_layer_i_y = LinearDCT(H, 'idct', drop=drop, ortho_constraint=ortho_constraint, ortho_freq=ortho_freq)
        self.linear_layer_i_x = LinearDCT(W, 'idct', drop=drop, ortho_constraint=ortho_constraint, ortho_freq=ortho_freq)

    def forward(self, x):
        X = self.linear_layer_i_y(x.transpose(-1, -2))
        recon_X = self.linear_layer_i_x(X.transpose(-1, -2))
        return recon_X

if __name__ == '__main__':
    # x = torch.Tensor(1000,4096)
    # x.normal_(0,1)
    # linear_dct = LinearDCT(4096, 'dct')
    # error = torch.abs(dct(x) - linear_dct(x))
    # assert error.max() < 1e-3, (error, error.max())
    # linear_idct = LinearDCT(4096, 'idct')
    # error = torch.abs(idct(x) - linear_idct(x))
    # assert error.max() < 1e-3, (error, error.max())

    # See how well it reconstructs
    # x2 = torch.randn(10, 1920, 1080)
    # Load image with shape (H,W,3) and convert to (3,H,W) for PyTorch
    # x2 = torchvision.io.read_image("data/bunny/0001.png").float() / 255.0
    # H = x2.shape[1]
    # W = x2.shape[2]
    # X = dct_2d(x2)
    # recon_X = idct_2d(X)
    # error = torch.abs(x2 - recon_X)
    # print(f"Reconstruction error: {error.mean()}")

    # drop = 0.75
    # linear_layer_x = LinearDCT(W, 'dct', drop=drop)
    # linear_layer_y = LinearDCT(H, 'dct', drop=drop)
    # X = linear_layer_x(x2)
    # print(f"shape after first transformation: {X.shape}")
    # X = linear_layer_y(X.transpose(-1, -2))
    # print(f"shape after second transformation: {X.shape}")
    # linear_layer_i_y = LinearDCT(H, 'idct', drop=drop)
    # linear_layer_i_x = LinearDCT(W, 'idct', drop=drop)
    # X = linear_layer_i_y(X)
    # print(f"shape after inverse first transformation: {X.shape}")
    # recon_X = linear_layer_i_x(X.transpose(-1, -2))
    # print(f"shape after inverse second transformation: {recon_X.shape}")
    # # recon_X = idct_2d(X.transpose(-1, -2))
    # error = torch.abs(x2 - recon_X)
    # print(f"Reconstruction error: {error.mean()}")
    # torchvision.utils.save_image(recon_X, "reconstructed.png")



    class SimpleNet(nn.Module):
        def __init__(self, W, H, drop=0.75, ortho_constraint=False, ortho_freq=1):
            super(SimpleNet, self).__init__()
            # First create the layers without constraints
            self.linear_layer_x = LinearDCT(W, 'dct', drop=drop, ortho_constraint=ortho_constraint, ortho_freq=ortho_freq)
            self.linear_layer_y = LinearDCT(H, 'dct', drop=drop, ortho_constraint=ortho_constraint, ortho_freq=ortho_freq)
            self.linear_layer_i_y = LinearDCT(H, 'idct', drop=drop, ortho_constraint=ortho_constraint, ortho_freq=ortho_freq)
            self.linear_layer_i_x = LinearDCT(W, 'idct', drop=drop, ortho_constraint=ortho_constraint, ortho_freq=ortho_freq)
            
        def forward(self, x):
            X = self.linear_layer_x(x)
            X = self.linear_layer_y(X.transpose(-1, -2))
            # print(f"shape after second transformation: {X.shape}")
            # import pdb; pdb.set_trace()
            X = self.linear_layer_i_y(X)
            recon_X = self.linear_layer_i_x(X.transpose(-1, -2))
            return recon_X

    img_data = torchvision.io.read_image("data/bunny/0001.png").float() / 255.0
    # img_data = torch.load("/mnt/tmp/embed_1.pth", map_location='cpu')
    img_data = img_data[0].cuda()
    # Calculate mean and std for each channel
    # mean = img 
    H = img_data.shape[-2]
    W = img_data.shape[-1]
    drop = 0.5
    net = SimpleNet(W, H, drop=drop, ortho_constraint=False, ortho_freq=10).cuda()
    # test_linear = LinearDCT(W, 'dct', drop=drop).cuda()
    # with torch.no_grad():
    #     test = test_linear(img_data)
    #     test2 = net.linear_layer_x(img_data)
    #     diff = test - test2
    #     print(f"Difference: {diff.abs().max()}")
    # import pdb; pdb.set_trace()
    num_epochs=1000
    optimizer = Lion(net.parameters(), weight_decay=0.)
    from hnerv_utils import loss_fn, adjust_lr, psnr_fn_single
    # train encoder and decoder

    total_params = sum(p.numel() for p in net.parameters())
    # Count parameters for forward DCT layers
    fwd_params = sum(p.numel() for name, p in net.named_parameters() if 'linear_layer_x.' in name or 'linear_layer_y.' in name)
    
    # Count parameters for inverse DCT layers  
    inv_params = sum(p.numel() for name, p in net.named_parameters() if 'linear_layer_i_x.' in name or 'linear_layer_i_y.' in name)
    
    print(f"Forward DCT parameters: {fwd_params/1e6:.2f}M")
    print(f"Inverse DCT parameters: {inv_params/1e6:.2f}M")
    print(f"Total number of parameters in the network: {total_params/1e6:.2f}M")


    for epoch in range(num_epochs):
        # Apply learning rate scheduler
        cur_epoch = epoch / num_epochs
        lr = adjust_lr(optimizer, cur_epoch, args=type('Args', (), {
            'lr': 0.00001,
            'lr_type': 'cosine_0.05_2_0.1'  # Warm up to 10% with cosine decay
        }))
        optimizer.zero_grad()
        recon_X = net(img_data)
        loss = loss_fn(recon_X, img_data, loss_type='L2')
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            psnr = psnr_fn_single(recon_X, img_data)
            print(f"Epoch {epoch}, lr: {lr:.2e}, loss: {loss.item():.6f}, psnr: {psnr.mean().item():.3f}")
            # torchvision.utils.save_image(recon_X, f"reconstructed_{epoch}.png")
            if epoch == 0:
                stacked_images = recon_X
            else:
                stacked_images = torch.cat([stacked_images, recon_X], dim=1)
    # torchvision.utils.save_image(stacked_images, f"stacked_reconstructed.png")


    


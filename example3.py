"""How to use ``foldNd``. A comparison with ``torch.nn.Fold``."""

# imports, make this example deterministic
import torch

import unfoldNd

torch.manual_seed(0)

# random output of an im2col operation
inputs = torch.ones(1, 3276800, 576, device='cuda')
output_size = (4, 760, 1320)

# other module hyperparameters
kernel_size = (4, 80, 80)
dilation = (1, 1, 1)
padding = (0, 20, 20)
stride = (4, 40, 40)

# both modules accept the same arguments and perform the same operation
# torch_module = torch.nn.Fold(
#     output_size, kernel_size, dilation=dilation, padding=padding, stride=stride
# )
lib_module = unfoldNd.FoldNd(
    output_size, kernel_size, padding=padding, stride=stride
)

# forward pass
# torch_outputs = torch_module(inputs)
try:
    lib_outputs = lib_module(inputs)
except Exception as e:
    print(e)
    import pdb; pdb.set_trace()
    lib_outputs = unfoldNd.foldNd(inputs, output_size=output_size, kernel_size=kernel_size, stride=stride)

import pdb; pdb.set_trace()

print(lib_outputs.shape)
print(lib_outputs.max())
print(lib_outputs.min())

# check
# if torch.allclose(torch_outputs, lib_outputs):
#     print("✔ Outputs of torch.nn.Fold and unfoldNd.FoldNd match.")
# else:
#     raise AssertionError("❌ Outputs don't match")
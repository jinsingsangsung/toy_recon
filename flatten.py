import os
import pickle
import numpy as np
import torch
from einops import rearrange
import torchvision

def zigzag_flatten(img):
    """Flatten a 2D array in zigzag order"""
    h, w = img.shape[-2:]
    indices = np.zeros(h * w, dtype=np.int64)
    idx = 0
    for i in range(h):
        if i % 2 == 0:  # Even rows go left to right
            for j in range(w):
                indices[i*w + j] = idx
                idx += 1
        else:  # Odd rows go right to left
            for j in range(w-1, -1, -1):
                indices[i*w + j] = idx
                idx += 1
    
    img = img.reshape(3, -1)
    return img[..., indices]

def main():
    # Create output directory if it doesn't exist
    out_dir = "data/cifar-100-python/test-1d"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load CIFAR data
    with open("data/cifar-100-python/test", "rb") as f:
        data = pickle.load(f, encoding="bytes")
    
    # Get images and reshape to [N,C,H,W]
    images = torch.from_numpy(data[b"data"]).reshape(-1, 3, 32, 32)
    
    # Flatten each image in zigzag order
    flattened = []
    for img in images:
        flat_img = zigzag_flatten(img)
        flattened.append(flat_img)

    # # Save first image and its flattened version for visualization
    # if len(flattened) > 0:
    #     orig_img = images[0]
    #     flat_img = flattened[0].reshape(3, -1)
    #     # Stack original and flattened vertically
    #     vis_img = torch.cat([
    #         orig_img,
    #         flat_img.reshape(3, 32, 32)  # Reshape flattened back to 2D for visualization
    #     ], dim=1)
        
    #     # Save visualization
    #     vis_img = vis_img.permute(1,2,0) # Convert to HWC format
    #     # vis_img = (vis_img * 255).byte()
    #     torchvision.utils.save_image(vis_img.permute(2,0,1).float()/255, 'flattened_vis.png')

    # Stack back into tensor
    flattened = torch.stack(flattened)
    import pdb; pdb.set_trace()
    
    # Save flattened data
    output = {b"data": flattened.numpy()}
    with open(os.path.join(out_dir, "test"), "wb") as f:
        pickle.dump(output, f)

if __name__ == "__main__":
    main()

# %%
from captum.attr import LayerIntegratedGradients
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/home/jr_buler/unet3d-xai/pytorch-3dunet/')
from pytorch3dunet.unet3d.model import UNet3D
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
import logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

'''
This file creates a simple 3D U-Net model and demonstrates how to use Captum to compute and visualize
integrated gradients for specified layer. Maxpool layers in the model are updated to have a kernel size of 1x3x3,
to imitate 2D pooling (to retain slice dimension in the intermediate feature maps).
The model is trained on synthetic data with high intensity regions.

Inplace call of activation functions was set to False (default was True) in the buildingblocks.py file
in the pytorch3dunet package.
'''


# update MaxPool3d layers
def update_maxpool3d_params(model, new_kernel_size, new_stride, new_padding):
    for name, module in model.named_children():
        if isinstance(module, nn.MaxPool3d):
            setattr(model, name, nn.MaxPool3d(new_kernel_size, new_stride, new_padding))
        elif len(list(module.children())) > 0:  # Recursively apply to nested modules
            update_maxpool3d_params(module, new_kernel_size, new_stride, new_padding)


# init the model
net = UNet3D(in_channels=1, out_channels=1, f_maps=32, final_sigmoid=True, layer_order='gcr',
             num_groups=8)
# New parameters for MaxPool3d layers
new_kernel_size, new_stride, new_padding = (1, 3, 3), (1, 2, 2), (0, 1, 1)

update_maxpool3d_params(net, new_kernel_size, new_stride, new_padding)
net.eval()
# input_tensor = torch.randn((1, 1, 20, 120, 120))
# input_tensor = torch.ones((1, 1, 20, 120, 120))
input_tensor = torch.zeros((1, 1, 20, 120, 120))
input_tensor[:,:,10,:,:] = torch.ones(input_tensor[:,:,10,:,:].shape)
input_tensor[:,:,15,50:70,:] = torch.zeros_like(input_tensor[:,:,15,50:70,:]) + 1
input_tensor[:,:,5,:,50:70] = torch.zeros_like(input_tensor[:,:,5,:,50:70]) + 1
slice_index = 17
thickness = 5
for h in range(input_tensor.shape[3]):  # Loop over height
    for w in range(input_tensor.shape[4]):  # Loop over width
        # Condition for a diagonal band (difference between height and width falls in range)
        if abs(h - w) < thickness:
            input_tensor[:, :, slice_index, h, w] = 1

logger.info(f"input_tensor shape: {input_tensor.shape}")
input_tensor.requires_grad = True
#%%
class Modified3DUNet(nn.Module):
    def __init__(self, original_model):
        super(Modified3DUNet, self).__init__()
        self.original_model = original_model
        logger.info(f"Wrapped model: {self.original_model}")

    def forward(self, x):
        # Run the original forward pass
        model_out = self.original_model(x)
        logger.info(f"model_out shape: {model_out.shape}")
        out_max = (model_out > 0.9).type(torch.LongTensor)  # Shape: batch x channel x slice x height x width >>> channel to tutaj na outpucie są klasy za sigmoidą (final_sigmoid = True (default))
        if device.type == 'cuda':
            out_max = out_max.cuda()
        selected_inds = torch.zeros_like(model_out).squeeze(1)  # Remove channel dimension (for scatter_)
        logger.info(f"selected_inds shape: {selected_inds.shape}")
        logger.info(f"model_out shape: {model_out.shape}")
        # Create a binary mask with 1 at predicted class locations
        selected_inds.scatter_(1, out_max.squeeze(1), 1)  # Scatter in dimension 1, using valid indices
        logger.info(f"selected_inds shape: {selected_inds.shape}")
        # Element-wise multiplication of model output and the binary mask
        # and sum over spatial dimensions (slice, height, width)
        logger.info(f"{(model_out * selected_inds).sum(dim=(2,3,4))=}")
        logger.info(f"{(model_out * selected_inds).sum(dim=(2, 3, 4)).shape=}")
        return (model_out * selected_inds).sum(dim=(2, 3, 4))  # [batch, num_classes]

unet = Modified3DUNet(net)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # gpu 24gb is not enough
device = torch.device("cpu")
logger.info(f"Device set to: {device}")
unet.to(device)
input_tensor = input_tensor.to(device)
# %%
# Generate synthetic training data

def generate_synthetic_data(num_samples, shape, high_intensity_value=1, low_intensity_value=0):
    data = []
    targets = []
    for _ in range(num_samples):
        sample = torch.zeros(shape)
        target = torch.zeros(shape)
        
        # Randomly place high intensity regions
        num_high_intensity_regions = np.random.randint(1, 5)
        for _ in range(num_high_intensity_regions):
            z = np.random.randint(0, shape[1])
            y = np.random.randint(0, shape[2])
            x = np.random.randint(0, shape[3])
            depth = np.random.randint(1, shape[1] // 4)
            height = np.random.randint(1, shape[2] // 4)
            width = np.random.randint(1, shape[3] // 4)
            
            sample[:, z:z+depth, y:y+height, x:x+width] = high_intensity_value
            target[:, z:z+depth, y:y+height, x:x+width] = 1
        
        data.append(sample)
        targets.append(target)
    
    return torch.stack(data), torch.stack(targets)
def train_unet(model, data, targets, num_epochs=5, learning_rate=0.001):
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(data)
        outputs = outputs - min(outputs.flatten())
        outputs = outputs / outputs.max()  # Normalize to [0, 1]
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
if 0:
# Generate 10 samples of shape (1, 20, 120, 120)
    num_samples = 10
    shape = (1, 20, 120, 120)
    data, targets = generate_synthetic_data(num_samples, shape)
    data = data.to(device)
    targets = targets.to(device)
    # Define a simple training loop
    # Train the model
    train_unet(unet.original_model, data, targets)
# %%
unet.eval()
target_layers = [encoder.basic_module[-1].conv for encoder in unet.original_model.encoders]



for i, target_layer in enumerate(target_layers):
    layer_= LayerIntegratedGradients(unet, target_layer)
    attribution = layer_.attribute(input_tensor, target=0)
    # attribution = attribution.mean(dim=1, keepdim=True)
    attribution, _ = attribution.max(dim=1, keepdim=True)

    nrows, ncols = 5, 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 10))

    # max_attr = attribution.flatten().max()
    # min_attr = attribution.flatten().min()

    for j in range(attribution.shape[2]):
        # Select the i-th slice along the depth axis
        slice_attr = attribution[0, 0, j, :, :].cpu().detach().numpy()
        slice_attr = np.expand_dims(slice_attr, axis=-1)
        logger.info(f"slice {j}:\t{min(slice_attr.flatten())}\t{max(slice_attr.flatten())}\t{slice_attr.flatten().mean()}")
        ROW, COL = j // ncols, j % ncols
        ax = axes[ROW, COL]
        axes[ROW, COL].imshow(slice_attr, cmap='coolwarm')#, vmin=min_attr, vmax=max_attr)
        axes[ROW, COL].axis('off')
        axes[ROW, COL].set_title(f'slice no. {j}')
    fig.suptitle(f"Encoder no. {i} last conv layer")
    plt.tight_layout()
    plt.show()
# %%
nrows, ncols = 5, 4
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 10))
for i in range(input_tensor.shape[2]):
    # Select the i-th slice along the depth axis
    slice_attr = input_tensor[0, 0, i, :, :].cpu().detach().numpy()
    slice_attr = np.expand_dims(slice_attr, axis=-1)
    row = i // ncols
    col = i % ncols
    ax = axes[row, col]
    axes[row, col].imshow(slice_attr, cmap='coolwarm', vmin=0, vmax=1)
    axes[row, col].axis('off')
    axes[row, col].set_title(f'slice no. {i}')
plt.tight_layout()
plt.show()
# %%
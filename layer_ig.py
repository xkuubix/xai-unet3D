# %%
from captum.attr import LayerIntegratedGradients
import sys
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
sys.path.append('/home/jr_buler/unet3d-xai/pytorch-3dunet/')
from pytorch3dunet.unet3d.model import UNet3D
from utils import *
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
import logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

'''
This file creates a simple 3D U-Net model and demonstrates how to use Captum to compute and visualize
integrated gradients. Maxpool layers in the model are updated to have a kernel size of 1x3x3,
to imitate 2D pooling (to retain slice dimension in the intermediate feature maps).
The model is pretrained WMH Challenge data.

Inplace call of activation functions was set to False (default was True) in the buildingblocks.py file
in the pytorch3dunet package.
'''

data_path = '/media/dysk_a/jr_buler/WMH/dataverse_files/test/'
data_instance = 'Amsterdam/Philips/160/'
mri_file_path = data_path + data_instance + 'pre/FLAIR.nii.gz'
mask_file_path = data_path + data_instance + 'wmh.nii.gz'

model_path = './final_best_checkpoint.pytorch'

# load model and update MaxPool3d layers
net = UNet3D(in_channels=1, out_channels=1, f_maps=32, final_sigmoid=True, layer_order='gcr',
             num_groups=8)
new_kernel_size, new_stride, new_padding = (1, 3, 3), (1, 2, 2), (0, 1, 1)
update_maxpool3d_params(net, new_kernel_size, new_stride, new_padding)
model_state_dict = torch.load(model_path,
                              map_location='cpu',
                              weights_only=True)['model_state_dict']
net.load_state_dict(model_state_dict)
net.eval()

# load brain  data
brain = {"image": np.asarray(nib.load(mri_file_path).dataobj),
         "mask" : np.asarray(nib.load(mask_file_path).dataobj)}
brain['image'] = torch.from_numpy(brain['image']).permute(2, 0, 1)
brain['mask'] = torch.from_numpy(brain['mask']).permute(2, 0, 1)

brain['mask'][brain['mask']==2] =  0# only WMH

#preprocessing - normalize image
brain['image'] = brain['image'] / brain['image'].max()

# get patches from the brain data: 20 x 120 x 120
x = 60
brain_patch      = brain['image'][40:60, x:x+120, x:x+120].unsqueeze(0).unsqueeze(0)
brain_mask_patch =  brain['mask'][40:60, x:x+120, x:x+120].unsqueeze(0).unsqueeze(0)
del brain
# show brain patch and mask
plot_slices(brain_patch, "Patch slices")
plot_slices(brain_mask_patch, "Ground truth")
logger.info(f"input_tensor shape: {brain_patch.shape}")
brain_patch.requires_grad = True
#%%
class ModelWrapper(nn.Module):
    def __init__(self, original_model):
        super(ModelWrapper, self).__init__()
        self.original_model = original_model
        logger.info(f"Wrapped model: {self.original_model}")

    def forward(self, x):
        # Run the original forward pass
        model_out = self.original_model(x)
        return model_out.sum(dim=(2, 3, 4))  # [batch, num_classes]
        logger.info(f"model_out shape: {model_out.shape}")
        out_max = (model_out > 0.6).type(torch.LongTensor)  # Shape: batch x channel x slice x height x width >>> channel to tutaj na outpucie są klasy za sigmoidą (final_sigmoid = True (default))
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
        logger.info(f"{(model_out * selected_inds).sum(dim=(2, 3, 4)).shape=}")
        return (model_out * selected_inds).sum(dim=(2, 3, 4))  # [batch, num_classes]

unet = ModelWrapper(net)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # gpu 24gb is not enough for such big patch
device = torch.device("cpu")
logger.info(f"Device set to: {device}")
unet.to(device)
brain_patch = brain_patch.to(device)
prediction = unet.original_model(brain_patch).detach()
prediction = (prediction > 0.6).type(torch.LongTensor)
# threshold prediction - round ==> th=0.5
plot_slices(prediction, "Prediction slices")
# %%
unet.eval()
target_layers = [encoder.basic_module[-1].conv for encoder in unet.original_model.encoders]
for i, target_layer in enumerate(target_layers):
    layer_= LayerIntegratedGradients(unet, target_layer, multiply_by_inputs=False)
    attribution = layer_.attribute(brain_patch, target=0)
    attribution_max, _ = attribution.max(dim=1, keepdim=True)
    absolute_attributions = torch.abs(attribution_max)
    flattened_attributions = absolute_attributions.view(-1).cpu().detach().numpy()
    percentile_99 = np.percentile(flattened_attributions, 99)
    capped_attributions = torch.clamp(absolute_attributions, max=percentile_99)
    min_val = capped_attributions.min()
    max_val = capped_attributions.max()
    normalized_attributions = (capped_attributions - min_val) / (max_val - min_val + 1e-10)
    plot_slices(normalized_attributions, f"layer no. {i} integrated gradients")
    print(normalized_attributions.shape)
print("Done")
# %%
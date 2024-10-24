import torch.nn as nn
import matplotlib.pyplot as plt

def update_maxpool3d_params(model, new_kernel_size, new_stride, new_padding):
    for name, module in model.named_children():
        if isinstance(module, nn.MaxPool3d):
            setattr(model, name, nn.MaxPool3d(new_kernel_size, new_stride, new_padding))
        elif len(list(module.children())) > 0:  # Recursively apply to nested modules
            update_maxpool3d_params(module, new_kernel_size, new_stride, new_padding)


def plot_slices(data, title, nrows=5, ncols=4, cmap='gray', vmin=None, vmax=None):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 10))
    for i in range(data.shape[2]):
        ROW = i // ncols
        COL = i % ncols
        ax = axes[ROW, COL]
        slice = data[0, 0, i, :, :].detach().numpy()
        if vmax and vmin:
            axes[ROW,COL].imshow(slice, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            axes[ROW,COL].imshow(slice, cmap=cmap)
        axes[ROW,COL].axis('off')
        axes[ROW,COL].set_title(f'slice no. {i}')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
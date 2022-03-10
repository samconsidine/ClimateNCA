import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from models import CAModel
from utils import ticktock
#from model import CAModel
from dataset import ClimateHackDataset
import xarray as xr
from loss import MS_SSIMLoss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = CAModel(num_channels=12, hidden_channels=128)
optimiser = optim.Adam(model.parameters(), lr=1e-1)
criterion = MS_SSIMLoss(channels=24) # produces less blurry images than nn.MSELoss()


SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"

dataset = xr.open_dataset(
    SATELLITE_ZARR_PATH, 
    engine="zarr",
    chunks="auto",  # Load the data as a Dask array
)


ch_dataset = ClimateHackDataset(dataset, crops_per_slice=1, day_limit=7)
ch_dataloader = DataLoader(ch_dataset, batch_size=1)

@ticktock
def generate_predictions(model, inputs, crop=True):
    state = inputs# .permute((0, 2, 3, 1))
    outputs = None

    if crop:
        crop_fn = lambda x: x[:,:1,32:96,32:96]
    else:
        crop_fn = lambda x: x[:,:1,:,:]

    for step in range(1, 24 + 1):
        state = model(state)
        if step % 1 == 0:
            if outputs is None:
                outputs = crop_fn(state)
            else:
                outputs = torch.cat((outputs, crop_fn(state)), dim=1)

    return outputs

x = dataset["data"].sel(time=slice("2020-07-01 12:00", "2020-07-01 12:55")).isel(x=slice(128, 256), y=slice(128, 256)).to_numpy()
# y = dataset["data"].sel(time=slice("2020-07-01 13:00", "2020-07-01 14:55")).isel(x=slice(160, 224), y=slice(160, 224)).to_numpy()
y = dataset["data"].sel(time=slice("2020-07-01 13:00", "2020-07-01 14:55")).isel(x=slice(128, 256), y=slice(128, 256)).to_numpy()

def plot_model_output():
    p = generate_predictions(model, torch.from_numpy(x.astype(np.float32)).unsqueeze(dim=0), crop=False).detach().numpy()[0]
    print(p.shape)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 12, figsize=(20,8))

    # plot the twelve 128x128 input images
    for i, img in enumerate(x):
        ax1[i].imshow(isolate_cloud_formations(img), cmap='Greys')
        ax1[i].get_xaxis().set_visible(False)
        ax1[i].get_yaxis().set_visible(False)

    # plot twelve 64x64 true output images
    for i, img in enumerate(y[:12]):
        ax2[i].imshow(isolate_cloud_formations(img), cmap='Greys')
        ax2[i].get_xaxis().set_visible(False)
        ax2[i].get_yaxis().set_visible(False)

    # plot twelve more 64x64 true output images
    for i, img in enumerate(y[12:]):
        ax3[i].imshow(isolate_cloud_formations(img), cmap='Greys')
        ax3[i].get_xaxis().set_visible(False)
        ax3[i].get_yaxis().set_visible(False)

    # plot the twelve 64x64 predicted output images
    for i, img in enumerate(p[:12]):
        ax4[i].imshow(isolate_cloud_formations(img), cmap='Greys')
        ax4[i].get_xaxis().set_visible(False)
        ax4[i].get_yaxis().set_visible(False)

    # plot twelve more 64x64 output images
    for i, img in enumerate(p[12:]):
        ax5[i].imshow(isolate_cloud_formations(img), cmap='Greys')
        ax5[i].get_xaxis().set_visible(False)
        ax5[i].get_yaxis().set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def isolate_cloud_formations(data):
    return (data > 100) * data

losses = []
for epoch in range(1):
    print(f"Epoch {epoch + 1}")
    running_loss = 0
    i = 0
    count = 0
    for batch_coordinates, batch_features, batch_targets in ch_dataloader:
        optimiser.zero_grad()

        batch_predictions = generate_predictions(model, isolate_cloud_formations(batch_features).to(device))
        print(f'{batch_predictions.shape=}')
        print(f'{batch_targets.shape=}')

        batch_loss = criterion(batch_predictions.unsqueeze(dim=2),
                isolate_cloud_formations(batch_targets).unsqueeze(dim=2).to(device))
        batch_loss.backward()

        optimiser.step()

        running_loss += batch_loss.item() * batch_predictions.shape[0]
        count += batch_predictions.shape[0]
        i += 1

        print(f"Completed batch {i} of epoch {epoch + 1} with loss {batch_loss.item()} -- processed {count} image sequences ({12 * count} images)")

    plot_model_output()
    losses.append(running_loss / count)
    print(f"Loss for epoch {epoch + 1}/{EPOCHS}: {losses[-1]}")




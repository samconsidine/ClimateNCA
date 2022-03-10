import numpy as np
import os
from glob import glob
import xarray as xr
from torch.utils.data import DataLoader

from dataset import ClimateHackDataset

if not os.path.exists('data'):
    os.mkdir('data')

SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"

dataset = xr.open_dataset(
    SATELLITE_ZARR_PATH, 
    engine="zarr",
    chunks="auto",  # Load the data as a Dask array
)

ch_dataset = ClimateHackDataset(dataset, crops_per_slice=1, day_limit=7)
ch_dataloader = DataLoader(ch_dataset, batch_size=64)

initialised = False

def mem_usage(arr: np.ndarray) -> int:
    return arr.size * arr.itemsize

def getsizeofnp(*arrays: np.ndarray) -> int:
    return sum(mem_usage(array) for array in arrays)

file_counter = 0

for batch_coordinates, batch_features, batch_targets in ch_dataloader:
    batch_coordinates = batch_coordinates.numpy()
    batch_features = batch_features.numpy()
    batch_targets = batch_targets.numpy()
    if not initialised:
        batch_coords_buffer = batch_coordinates
        batch_features_buffer = batch_features
        batch_targets_buffer = batch_targets
        initialised = True

    else:
        try:
            batch_coords_buffer = np.concatenate((batch_coords_buffer, batch_coordinates), axis=0)
            batch_features_buffer = np.concatenate((batch_features_buffer, batch_features), axis=0)
            batch_targets_buffer = np.concatenate((batch_targets_buffer, batch_targets), axis=0)
        except ValueError:
            print(f'{batch_coords_buffer.shape=}')
            print(f'{batch_coordinates.shape=}')

    size = getsizeofnp(batch_coords_buffer, batch_features_buffer, batch_targets_buffer)
    if size > 5_000_000_000:
        print('saving file...')
        np.savez(
            f'data/slice_{file_counter}.npz',
            batch_coordinates=batch_coords_buffer, 
            batch_features=batch_features_buffer,
            batch_targets=batch_targets_buffer
        )
        initialised = False
        file_counter += 1


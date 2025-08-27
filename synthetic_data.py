# Generate a synthetic zarr dataset

import dask.array as da
import zarr.storage

from dask.diagnostics import ProgressBar
from tqdm import tqdm

NPOSITIONS = 8640*4
NFEATURES = 1024
NTIMESTEPS = 12*32
TIMESTEP_CHUNK_SIZE = 1

def main():

  state = da.random.RandomState(0)

  make_chunk = lambda: state.random(size=(TIMESTEP_CHUNK_SIZE, NPOSITIONS, NFEATURES), chunks=(TIMESTEP_CHUNK_SIZE, 8640, 1)).compute().astype("float32")

  with ProgressBar():
    a = make_chunk()
    z = zarr.create_array(store='dataset.zarr', shape=a.shape, dtype=a.dtype, chunks=(1, 8640, 1))
    z[:] = a

  for i in tqdm(range(12*32)):
    a = make_chunk()
    z.append(a)

if __name__ == "__main__":
  main()
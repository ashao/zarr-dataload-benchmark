# Load data from a zarr datastore with some configurable options

import argparse

import torch
import zarr

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

class ZarrDataset(Dataset):
  def __init__(self, zarr_path):
    self.array = zarr.open(zarr_path, mode="r")
    self.ntime, self.npos, self.nfeatures = self.array.shape

  def __len__(self):
    return self.ntime

  def __getitem__(self, idx):
    return torch.tensor(self.array[idx])

def main(args):

  if args.distributed:
    torch.distributed.init_process_group()

  dataset = ZarrDataset(args.zarr_path)
  sampler = DistributedSampler(dataset) if args.distributed else None

  dataloader = DataLoader(
    dataset,
    sampler=sampler,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
    shuffle=False
  )

  for batch in tqdm(dataloader):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Zarr DataLoader benchmark with configurable parameters")
    parser.add_argument("--zarr-path", type=str, required=True, help="Path to the Zarr array")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pin_memory for DataLoader")
    parser.add_argument("--distributed", action="store_true", help="Enable torch distributed")

    args = parser.parse_args()
    main(args)
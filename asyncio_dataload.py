# Load data from a zarr using asyncio

import asyncio
import zarr
import time

zarr_path = "dataset.zarr"
zarr_array = zarr.open(zarr_path, mode='r')

ntime, npositions, nfeatures = zarr_array.shape
chunk_size = 16

async def load_chunk(start_idx):
    end_idx = min(start_idx + chunk_size, ntime)
    return await asyncio.to_thread(zarr_array.__getitem__, slice(start_idx, end_idx))

async def load_all_chunks():
    tasks = []
    for i in range(0, ntime, chunk_size):
        tasks.append(load_chunk(i))
    chunks = await asyncio.gather(*tasks)
    return chunks

if __name__ == "__main__":
    start = time.time()
    chunks = asyncio.run(load_all_chunks())
    print(f"Loaded {len(chunks)} chunks. in {time.time() - start}s")

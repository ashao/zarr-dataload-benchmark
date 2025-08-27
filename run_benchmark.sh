#!/bin/bash

if [ ! -d "dataset.zarr" ]; then
  python synthetic_data.py
fi

python asyncio_dataload.py
python torch_dataload.py --zarr-path dataset.zarr --batch_size=16 --num_workers=16
python torch_dataload.py --zarr-path dataset.zarr --batch_size=16 --num_workers=32
torchrun --nproc_per_node=4 torch_dataload.py \
                            --zarr-path dataset.zarr \
                            --batch_size=16 \
                            --num_workers=4 \
                            --distributed
torchrun --nproc_per_node=4 torch_dataload.py \
                            --zarr-path dataset.zarr \
                            --batch_size=16 \
                            --num_workers=8 \
                            --distributed
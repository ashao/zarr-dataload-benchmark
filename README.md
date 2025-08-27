# Summary
The scripts here create a directory-backed zarr dataset and various ways of
loading the data back in including 1) an asyncio implementation and 2) using
torch dataloaders with configurable batch size and workers and 3) using torch
data distributed parallel sampler (with torchrun).

# Results
On a machine with 128 cores, with the zarr array stored on a lustre filesystem
the following times were captured for a batchsize of 16

- asyncio: 15m 50s
- torch dataloader (1 worker): 16m 32s
- torch dataloader (16 workers): 2m 4s
- torch dataloader (32 workers): 1m 15s
- torch DDP dataloader (4 procs, 4 workers each): 1m 52s
- torch DDP dataloader (4 procs, 8 workers each): 1m 10s
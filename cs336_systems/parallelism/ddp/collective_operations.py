import torch
import torch.distributed as dist

from cs336_systems.parallelism.utils.utils import setup, cleanup, get_device, spawn


def collective_operations_main(rank: int, world_size: int):
  """This function is running asynchronously for each process (rank = 0, ..., world_size - 1)."""
  setup(rank, world_size)

  # All-reduce
  dist.barrier()  # Waits for all processes to get to this point (in this case, for print statements)

  tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank  # Both input and output

  print(f"Rank {rank} [before all-reduce]: {tensor}", flush=True)
  dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
  print(f"Rank {rank} [after all-reduce]: {tensor}", flush=True)

  # Reduce-scatter
  dist.barrier()

  input = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank  # Input
  output = torch.empty(1, device=get_device(rank))  # Allocate output

  print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
  dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
  print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)

  # All-gather
  dist.barrier()

  input = output  # Input is the output of reduce-scatter
  output = torch.empty(world_size, device=get_device(rank))  # Allocate output

  print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
  dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
  print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)
  
  cleanup()

if __name__ == "__main__":
  spawn(collective_operations_main, world_size=2)
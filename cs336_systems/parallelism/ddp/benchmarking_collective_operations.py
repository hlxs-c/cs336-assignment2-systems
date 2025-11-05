import time
import torch
import torch.distributed as dist

from cs336_systems.parallelism.utils.utils import spawn, setup, cleanup, get_device, render_duration

def all_reduce(rank: int, world_size: int, num_elements: int):
  """
  这是一个用于测量分布式 all_reduce 操作性能的函数. 
  它会在多个进程 (由rank和world_size指定) 上执行 all_reduce 操作, 并计算带宽.

  Args:
    rank (int): 当前进程的id
    world_size (int): 进程总数
    num_elements (int): 要测试的张量元素数量
  """
  setup(rank=rank, world_size=world_size)

  # Create tensor
  tensor = torch.randn(num_elements, device=get_device(rank))

  # Warmup
  dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
  if torch.cuda.is_available():
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the process to get there
  
  # Perform all-reduce
  start_time = time.time()
  dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
  if torch.cuda.is_available():
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the process to get here
  end_time = time.time()

  duration = end_time - start_time
  print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took for {render_duration(duration=duration)}", flush=True)

  # Measure the effective bandwidth
  dist.barrier()
  size_bytes = tensor.element_size() * tensor.numel()
  send_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send input and receive output
  total_duration = world_size * duration
  bandwidth = send_bytes / total_duration
  print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

  cleanup()

def reduce_scatter(rank: int, world_size: int, num_elements: int):
  setup(rank=rank, world_size=world_size)

  # Create input and output
  input = torch.randn(world_size, num_elements, device=get_device(rank))  # Each rank has a matrix
  output = torch.empty(num_elements, device=get_device(rank))

  # Warmup
  dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
  if torch.cuda.is_available():
    torch.cuda.synchronize()  # Wait for CUDA kernel to finish
    dist.barrier()            # Wait for all process to get here

  # Perform reduce-scatter
  start_time = time.time()
  dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
  if torch.cuda.is_available():
    torch.cuda.synchronize()  # Wait for CUDA kernel to finish
    dist.barrier()            # Wait for all process to get here
  end_time = time.time()

  duration = end_time - start_time
  print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

  # Measure the effective bandwidth
  dist.barrier()
  data_bytes = output.element_size() * output.numel()  # How much data in the output
  sent_bytes = data_bytes * (world_size - 1)  # How much needs to be sent (no 2x here)
  total_duration = world_size * duration  # Total time for transmission
  bandwidth = sent_bytes / total_duration
  print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

  cleanup()



def benchmarking():
  # All-reduce
  spawn(all_reduce, world_size=2, num_elements=100 * 1024**2)

  # Reduce-scatter
  spawn(reduce_scatter, world_size=2, num_elements=100 * 1024**2)

if __name__ == "__main__":
  benchmarking()

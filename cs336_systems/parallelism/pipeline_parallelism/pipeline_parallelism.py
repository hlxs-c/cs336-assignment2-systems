import torch
import torch.nn.functional as F
import torch.distributed as dist

from cs336_systems.parallelism.utils.data_utils import generate_sample_data
from cs336_systems.parallelism.utils.utils import spawn, setup, cleanup, get_device, int_divide, get_init_params, summarize_tensor

def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
  setup(rank, world_size)

  # Use all the data
  data = data.to(get_device(rank))
  batch_size = data.size(0)
  num_dim = data.size(1)

  # Split up layers
  local_num_layers = int_divide(num_layers, world_size)

  # Each rank gets a subset of layers
  local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]
  
  # Break up into micro batches to minimize the bubble
  micro_batch_size = int_divide(batch_size, num_micro_batches)  # @inspect micro_batch_size
  if rank == 0:
    # The data
    # 第一个进程（主进程）处理的是原始数据输入，所以主进程需要将数据划分为 num_micro_batches 个微批次数据
    micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
  else:
    # Allocate memory for activations
    # 其他进程处理的是上一个进程产生的激活值，每一个进程都会产生 num_micro_batches 个微批次激活值，需要开辟空间接收
    micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) for _ in range(num_micro_batches)]
  
  # Forward pass
  for x in micro_batches:
    # Get activations from previous rank
    # 等待接收上一个进程的该微批次的激活值
    if rank - 1 >= 0:
      dist.recv(tensor=x, src=rank - 1)

    # Compute layers assigned to this rank
    # 使用该进程的模型层对该微批次进行前向计算
    for param in local_params:
      x = x @ param
      x = F.gelu(x)

    # Send to the next rank
    # 将该进程计算完成后的该微批次的激活值发送给下一个进程
    if rank + 1 < world_size:
      print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
      dist.send(tensor=x, dst=rank + 1)
  
  # Backward pass: homework exercise

  cleanup()


if __name__ == "__main__":
  data = generate_sample_data()
  spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)
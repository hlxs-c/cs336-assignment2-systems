import torch
import torch.nn.functional as F
import torch.distributed as dist

from cs336_systems.parallelism.utils.utils import spawn, setup, cleanup, int_divide, get_device, get_init_params, summarize_tensor
from cs336_systems.parallelism.utils.data_utils import generate_sample_data

def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
  setup(rank=rank, world_size=world_size)

  # Get the slice data for this rank (in practice, each rank should load only its own data)
  batch_size = data.size(0)
  num_dim = data.size(1)

  local_batch_size = int_divide(a=batch_size, b=world_size) # 每个进程分配的批次大小
  start_index = rank * local_batch_size                     # 当前进程的数据开始位置
  end_index = start_index + local_batch_size                # 当前进程的数据结束位置
  data = data[start_index:end_index].to(get_device(rank))

  # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
  params = [get_init_params(num_inputs=num_dim, num_outputs=num_dim, rank=rank) for i in range(num_layers)]
  optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank process optimizer state

  for step in range(num_steps):
    # Forward pass
    x = data
    for param in params:
      x = x @ param
      x = F.gelu(x)
    loss = x.square().mean()  # Loss function is average squared magnitude

    # Backward pass
    loss.backward()

    # Sync gradients across workers (only difference between standard training and DDP)
    for param in params:
      dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
    
    # Update parameters
    optimizer.step()

    print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", flush=True)
  
  cleanup()

if __name__ == "__main__":
  data = generate_sample_data(batch_size=128, num_dim=1024)
  spawn(data_parallelism_main, world_size=2, data=data, num_layers=4, num_steps=4)
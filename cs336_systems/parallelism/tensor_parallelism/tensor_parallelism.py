import torch
import torch.nn.functional as F
import torch.distributed as dist

from cs336_systems.parallelism.utils.data_utils import generate_sample_data
from cs336_systems.parallelism.utils.utils import spawn, setup, cleanup, get_device, int_divide, get_init_params, summarize_tensor

def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
  setup(rank=rank, world_size=world_size)

  data = data.to(device=get_device(rank))
  batch_size = data.size(0)
  num_dim = data.size(1)

  # shard `num_dim`
  local_num_dim = int_divide(a=num_dim, b=world_size)

  # Create model (each rank gets 1/world_size of the parameters)
  params = [get_init_params(num_inputs=num_dim, num_outputs=local_num_dim, rank=rank) for i in range(num_layers)]

  # Forward pass
  x = data
  for i in range(num_layers):
    # Compute activations (batch_size x local_num_dim)
    x = x @ params[i] # Note: this is only on a slice of the parameters
    x = F.gelu(x)

    # Allocate memory for activations (world_size x batch_size x local_num_dim)
    activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) for _ in range(world_size)]

    # Send activations via all gather
    dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

    # Concatenate them to get batch_size x num_dim
    x = torch.cat(activations, dim=1)
  
  print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)

  # Backward pass: homework exercise

  cleanup()

if __name__ == "__main__":
  data = generate_sample_data()
  spawn(tensor_parallelism_main, world_size=2, data=data, num_layers=4)

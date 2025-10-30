import os
import torch
import re

from torch.profiler import ProfilerActivity
from typing import Callable

def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False, record_shapes: bool = True, profile_memory: bool = True):
  """
  Profile the execution of a function using Pytorch Profiler.

  Args:
    description (str): Description of the profiling run
    run (Callable): Function to profile
    num_warmups (int): Number of warmup runs before profiling, default is 1
    with_stack (bool): Whether to capture stack traces, default is False
    record_shapes (bool): Whether to record tensor shape, default is False
    profile_memory (bool): Whether to profile memory, default is False
  """
  # Warmup runs
  for _ in range(num_warmups):
    run()
  
  # Synchronize if CUDA is available
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  
  try:
    # Profile execution
    with torch.profiler.profile(
      activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
      with_stack=with_stack,
      record_shapes=record_shapes, # recored tensor shapes
      profile_memory=profile_memory # profile memory usage
    ) as prof:
      run()

      if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Print profiling results table
    print(f'\n{'='*50} Profiling Results: {description} {'='*50}')
    table = prof.key_averages().table(
      sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
      max_name_column_width=80,
      row_limit=15
    )
    print(table)

    # Export stacks if requested
    if with_stack:
      os.makedirs("profile_tmp", exist_ok=True)

      # Sanitize description for filename
      safe_description = re.sub(r'[^\w\-_.]', '_', description)

      # Export stack traces
      text_path = os.path.join("profile_tmp", f"stacks_{safe_description}.txt")
      prof.export_stacks(text_path, "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total")
      print(f"Stack traces exported to: {text_path}")

    return table
  except Exception as e:
    print(f"Profiling failed: {e}")
    return None
  
if __name__ == "__main__":
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f'device: {device}')
  def matrix_multi():
    mat1 = torch.rand(128, 128, dtype=torch.float32, device=device)
    mat2 = torch.rand(128, 128, dtype=torch.float32, device=device)
    return mat1 @ mat2
  
  table = profile(description="matrix_multi", run=matrix_multi, profile_memory=False)
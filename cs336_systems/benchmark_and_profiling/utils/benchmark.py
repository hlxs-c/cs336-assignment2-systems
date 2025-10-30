import torch
import numpy as np
import time
from typing import Callable

def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
  """
  Benchmark `func` by running it `num_trials`, and return all the times.
  Args:
    description (str): Test description information.
    run (Callable): The callable function to be tested.
    num_warmups (int): Warups iters, default is 1
    num_trails (int): Actual testing iters, default is 3
  """
  # Warmup: first times might be slower due to compilation, things not cached.
  # Since we will run the kernel multiple times, the timing that matters is steady state.
  for _ in range(num_warmups):
    run()
  
  if torch.cuda.is_available():
    torch.cuda.synchronize()  # Wait for CUDA threads to finish
  
  # Time it for real now!
  times: list[float] = []
  for trial in range(num_trials):
    start_time = time.time()

    # Actually perform compuation
    run()

    if torch.cuda.is_available():
      torch.cuda.synchronize()  # Wait for CUDA threads to finish
    
    end_time = time.time()
    times.append((end_time - start_time) * 1000)  # convert time to millseconds
  
  mean_time = np.mean(times)
  return mean_time

if __name__ == "__main__":
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f'device: {device}')
  def matrix_multi():
    mat1 = torch.rand(128, 128, dtype=torch.float32, device=device)
    mat2 = torch.rand(128, 128, dtype=torch.float32, device=device)
    return mat1 @ mat2
  
  res = benchmark(description="matrix_multi", run=matrix_multi, num_warmups=3, num_trials=10)
  print(f'matrix_multi (128, 128) x (128, 128) cost: {res} millseconds')
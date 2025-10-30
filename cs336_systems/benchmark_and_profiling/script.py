import argparse
import torch
import torch.nn as nn
import time
import numpy as np

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark_and_profiling.utils.benchmark import benchmark

def benchmark_main_with_backward(
  description: str, 
  num_warmups: int, 
  num_trials: int, 
  model: nn.Module, 
  batch_x: torch.Tensor, 
  batch_y: torch.Tensor
) -> tuple[float, float, float]:
  """
  Benchmark model with forward and backward passes.
  """
  optimizer = AdamW(params=model.parameters())
  
  # Warmup
  for _ in range(num_warmups):
    logits = model(batch_x)
    loss = cross_entropy(logits, batch_y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
  
  if torch.cuda.is_available():
    torch.cuda.synchronize()  # Wait for CUDA
  
  # Benchmark
  forward_times: list[float] = []
  backward_times: list[float] = []

  for iter in range(num_trials):
    # Forward pass (including loss calculation)
    forward_start = time.time()
    logits = model(batch_x)
    loss = cross_entropy(logits, batch_y)
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    forward_end = time.time()

    # Backward pass
    backward_start = time.time()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    backward_end = time.time()

    forward_time = (forward_end - forward_start) * 1000
    backward_time = (backward_end - backward_start) * 1000
    forward_times.append(forward_time)
    backward_times.append(backward_time)
  
  forward_avg_time = np.mean(forward_times)
  backward_avg_time = np.mean(backward_times)
  total_avg_time = forward_avg_time + backward_avg_time

  return total_avg_time, forward_avg_time, backward_avg_time

    
def benchmark_main_without_backward(
  description: str, 
  num_warmups: int, 
  num_trials: int, 
  model: nn.Module, 
  batch_x: torch.Tensor, 
  batch_y: torch.Tensor
) -> float:
  """
  Benchmark model with only forward pass.
  """
  def run_transformerlm_forward():
    logits = model(batch_x)
    loss = cross_entropy(logits, batch_y)
  
  forward_avg_time = benchmark(
    description=description, 
    run=run_transformerlm_forward, 
    num_warmups=num_warmups, 
    num_trials=num_trials
  )
  
  return forward_avg_time


def get_args():
  parser = argparse.ArgumentParser()

  # 模型配置参数
  parser.add_argument("--context_length", type=int, default=256, help="context length")
  parser.add_argument("--num_layers", type=int, default=12, help="num layers of transformer block")
  parser.add_argument("--d_model", type=int, default=768, help="dimension of the transformer block")
  parser.add_argument("--num_heads", type=int, default=12, help="num heads of multi-head self-attention")
  parser.add_argument("--d_ff", type=int, default=3072, help="hidden dimensions of feed-forward network")
  parser.add_argument("--rope_theta", type=float, default=10000.0, help="base theta for rope")

  # benchmakr 相关参数
  parser.add_argument("--only_forward", type=bool, default=False, help="whether profile backward")
  parser.add_argument("--num_warmups", type=int, default=5, help="number of warmups before real benchmark")
  parser.add_argument("--num_trials", type=int, default=10, help="number of trails for real benchmark")

  args = parser.parse_args()

  return args

if __name__ == "__main__":
  args = get_args()
  print(f"Benchmark configuration: \n {args}")

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device: {device}")

  # 实例化 transformerlm 模型
  model_configuration = {
    "vocab_size": 10000,
    "context_length": args.context_length,
    "num_layers": args.num_layers,
    "d_model": args.d_model,
    "num_heads": args.num_heads,
    "d_ff": args.d_ff,
    "rope_theta": args.rope_theta,
  }
  model = BasicsTransformerLM(**model_configuration).to(device)

  # 打印模型参数数量
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Model parameters: {total_params}")

  # 生成随机批次数据
  batch_x = torch.randint(
    low=0, 
    high=10000, 
    size=(4, args.context_length)
  ) # shape: (batch_size, seq_length) = (4, context_length)
  batch_y = torch.cat(
    (batch_x[:, 1:], torch.randint(low=0, high=10000, size=(4, 1))), 
    dim=-1
  ) # shape: (batch_size, seq_length) = (4, context_length)
  
  # 将数据移动到设备
  batch_x = batch_x.to(device=device)
  batch_y = batch_y.to(device=device)

  print(f"Input shape: {batch_x.shape}")
  print(f"Target shape: {batch_y.shape}")

  if args.only_forward:
    avg_time = benchmark_main_without_backward(
      description="benchmark_transformerlm_without_backward",
      num_warmups=args.num_warmups,
      num_trials=args.num_trials,
      model=model,
      batch_x=batch_x,
      batch_y=batch_y
    )
    print(f'{"="*20} benchmark_transformerlm_without_backward {"="*20}')
    print(f'avg_time(avg_forward_time): {avg_time} milliseconds')
  else:
    total_avg_time, forward_avg_time, backward_avg_time = benchmark_main_with_backward(
      description="benchmark_transformerlm_with_backward",
      num_warmups=args.num_warmups,
      num_trials=args.num_trials,
      model=model,
      batch_x=batch_x,
      batch_y=batch_y
    )
    print(f'{"="*20} benchmark_transformerlm_with_backward {"="*20}')
    print(f'total_avg_time: {total_avg_time} milliseconds')
    print(f'forward_avg_time: {forward_avg_time} milliseconds')
    print(f'backward_avg_time: {backward_avg_time} milliseconds')

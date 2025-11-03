import torch
import torch.nn as nn
import time
import math
import numpy as np
import pandas as pd

from itertools import product

from cs336_basics.model import scaled_dot_product_attention

def benchmark_attention(
  num_warmups: int,
  num_trials: int,
  batch_size: int, 
  seq_len: int, 
  d_model: int, 
  device: torch.device
):
  try:
    # Clear GPU memory
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
      torch.cuda.synchronize()
    
    # Create random inputs
    Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    # Warmup
    for _ in range(num_warmups):
      scaled_dot_product_attention(Q=Q, K=K, V=V)
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    
    # Benchmark
    forward_times: list[float] = []
    backward_times: list[float] = []
    memory_usages: list[float] = []
    for iter in range(num_trials):
      # 1. 在每次独立测试前，确保状态干净（清除之前迭代可能留下的梯度）
      if Q.grad is not None:
        Q.grad.zero_()
      if K.grad is not None:
        K.grad.zero_()
      if V.grad is not None:
        V.grad.zero_()
      
      # 确保所有清理操作完成
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      
      # ---- 前向传播计时 -- 
      forward_start = time.time()
      output = scaled_dot_product_attention(Q=Q, K=K, V=V)
      loss = output.sum()
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      forward_end = time.time()

      # 2. 在一个干净的状态下测量内存，确保我们只测量当前前向传播的激活内存
      if torch.cuda.is_available():
        mem_before_backward = torch.cuda.memory_allocated() / (1024**2)
        memory_usages.append(mem_before_backward)

      # ---- 反向传播计时 ----
      backward_start = time.time()
      loss.backward()
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      backward_end = time.time()

      # times
      forward_time = (forward_end - forward_start) * 1000 # Convert to ms
      backward_time = (backward_end - backward_start) * 1000  # Convert to ms
      forward_times.append(forward_time)
      backward_times.append(backward_time)
    
    return {
      "forward_time_ms": np.mean(forward_times),
      "backward_time_ms": np.mean(backward_times),
      "memory_before_backward_mb": np.mean(memory_usages),
      "status": "Success"
    }
  except RuntimeError as e:
    if "out of memory" in str(e):
      print("\tOut of Memory Error")
      return {
      "forward_time_ms": float("nan"),
      "backward_time_ms": float("nan"),
      "memory_before_backward_mb": float("nan"),
      "status": "OOM"
    }
      

def benchmark_attention_main(num_warmups: int, num_trials: int):
  torch.manual_seed(42)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device: {device}")

  batch_size = 8
  d_models = [16, 32, 64, 128]
  seq_lengths = [256, 1024, 4096, 8192, 16384]

  results = []

  for d_model, seq_len in product(d_models, seq_lengths):
    print(f'Benchmarking d_model={d_model}, seq_len={seq_len}')

    benchmark_result = benchmark_attention(
      num_warmups=10,
      num_trials=100,
      batch_size=batch_size,
      seq_len=seq_len,
      d_model=d_model,
      device=device
    )

    result = {
      "d_model": d_model,
      "seq_len": seq_len,
      **benchmark_result
    }


    results.append(result)
    print(f"\tForward_avg_time: {result["forward_time_ms"]:.2f} ms, Backward_avg_time: {result["backward_time_ms"]:.2f} ms, memory_before_backward_mb: {result["memory_before_backward_mb"]:.2f} MB")
  
  return results

def calculate_memory(d_model: int, seq_len: int, batch_size: int = 8):
  """
  Calculate theoretical memory usage for attention.
  """
  # Input tensors: Q, K, V each or size [batch_size, seq_len, d_model]
  input_memory = 3 * batch_size * seq_len * d_model * 4   # 4 bytes per float32

  # Attention scores: [batch_size, seq_len, seq_len]
  scores_memory = batch_size * seq_len * seq_len * 4

  # Attention Weights: [batch_size, seq_len, seq_len]
  weights_memory = batch_size * seq_len * seq_len * 4  

  # Output: [batch_size, seq_len, d_model]
  output_memory = batch_size * seq_len * d_model * 4

  # Total forward pass memory (approximate)
  forward_memory = input_memory + scores_memory + weights_memory + output_memory

  return {
    "input_memory": input_memory / (1024**2),
    "scores_memory": scores_memory / (1024**2),
    "weights_memory": weights_memory / (1024**2),
    "output_memory": output_memory / (1024**2),
    "forward_total_memory": forward_memory / (1024**2)
  }


def main():
  print("Starting attention benchmark")
  results = benchmark_attention_main(num_warmups=10, num_trials=100)

  # Create results table
  df = pd.DataFrame(results)
  print("\n" + "="*80)
  print("RESULTS TABLE:")
  print("="*80)
  print(df.to_string())

  # Analyze memory usage for a configuration that runs OOM
  print("\n" + "="*80)
  print("MEMORY USAGE ANALYSIS:")
  print("="*80)

  # Find the smallest configuration that caused OOM
  oom_configs = [r for r in results if r["status"] == "OOM"]
  if oom_configs:
    # Take the smallest OOM configuration
    oom_config = min(oom_configs, key=lambda x: (x['d_model'], x['seq_len']))
    d_model = oom_config["d_model"]
    seq_len = oom_config["seq_len"]

    print(f"Analyzing memory usage for OOM configuration: d_model={d_model}, seq_len={seq_len}")
    memory_calc = calculate_memory(d_model=d_model, seq_len=seq_len)

    for key, value in memory_calc.items():
      print(f"{key}: {value:.2f} MB")
    
    # Compare with successful configurations
    success_configs = [r for r in results if r['status'] == 'Success']
    if success_configs:
      # Find largest successful configuration
      success_config = max(success_configs, key=lambda x: (x['d_model'], x['seq_len']))
      print(f"\nLargest successful configuration: d_model={success_config['d_model']}, seq_len={success_config['seq_len']}")
      print(f"Memory usage: {success_config['memory_before_backward_mb']:.2f} MB")

      memory_calc = calculate_memory(d_model=success_config['d_model'], seq_len=success_config['seq_len'])
      for key, value in memory_calc.items():
        print(f"{key}: {value:.2f} MB")

if __name__ == "__main__":
  main()

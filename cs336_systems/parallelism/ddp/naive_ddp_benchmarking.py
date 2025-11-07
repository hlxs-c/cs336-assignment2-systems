import time
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from cs336_systems.parallelism.utils.utils import setup, cleanup, spawn, get_device, int_divide, summarize_tensor

def broadcast_model(model: nn.Module, src_rank: int = 0):
  for param in model.parameters():
    dist.broadcast(param, src=src_rank)

def all_reduce_grad(model: nn.Module):
  for param in model.parameters():
    if param.grad is not None:
      dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)

def data_parallelism_main_with_benchmarking(
  rank: int, 
  world_size: int, 
  batch_x: torch.Tensor, 
  batch_y: torch.Tensor,
  num_warmups: int,
  num_trials: int,
  model_configuration: dict
):
  # 1. 初始化分布式环境（初始化当前进程）
  setup(rank=rank, world_size=world_size)

  # 2. 初始化模型和优化器
  # 实例化模型 （每个进程都要实例化一个模型）
  model = BasicsTransformerLM(**model_configuration)
  # 将模型移动到对应的设备
  model = model.to(device=get_device(rank))
  # 由主进程将模型参数广播给其他进程（保证所有GPU上的模型参数是一致的）
  for param in model.parameters():
    dist.broadcast(param.data, src=0)
  # 创建优化器 （由于之前已经保证了参数一致性，所以每个进程独立创建自己的优化器就能够保证所有进程的优化器状态一致）
  # optimizer = AdamW(model.parameters())
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.1) # lr 大一点才能看到参数的变化

  # 3. 切分数据（每个进程获得自己对应的数据）
  # 获取原始数据的元信息
  batch_size, seq_len = batch_x.shape
  # 计算每个进程的数据批次大小
  local_batch_size = int_divide(a=batch_size, b=world_size)
  # 计算每个进程的数据的位置
  local_start_index = rank * local_batch_size
  local_end_index = local_start_index + local_batch_size
  # 从原始数据中获得当前进程的批次数据
  local_batch_x = batch_x[local_start_index:local_end_index].to(device=get_device(rank))
  local_batch_y = batch_y[local_start_index:local_end_index].to(device=get_device(rank))

  print(f'[data_parallelism], Rank: {rank}, local_batch_x.shape: {local_batch_x.shape}')

  # 4. 预热阶段（Warmup）
  for _ in range(num_warmups):
    logits = model(local_batch_x)
    loss = cross_entropy(logits, local_batch_y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # 梯度同步
    for param in model.parameters():
      if param.grad is not None:
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
    optimizer.step()

  if torch.cuda.is_available():
    torch.cuda.synchronize()    # Wait for CUDA kernel finish
    dist.barrier()              # Wait for all the process to get here

  # 5. 基准测试（Benchmarking）
  total_times: list[float] = []
  communicating_grad_times: list[float] = []
  for step in range(num_trials):
    start_time = time.time()
    # 前向传播
    # 每个进程独立使用局部数据进行前向传播计算以及损失计算
    logits = model(local_batch_x)
    loss = cross_entropy(logits, local_batch_y)

    # 反向传播
    # 每个进程先清零对应梯度，然后用局部损失进行反向传播得到局部梯度
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    start_communicating_grad_time = time.time()
    # 通过 All-reduce 将每个GPU上的梯度平均得到全局平均梯度
    for param in model.parameters():
      if param.grad is not None:
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
    
    if torch.cuda.is_available():
      torch.cuda.synchronize()
      dist.barrier()
    end_communicating_grad_time = time.time()

    # 使用全局平均梯度进行参数更新
    optimizer.step()

    if torch.cuda.is_available():
      torch.cuda.synchronize()
      dist.barrier()

    end_time = time.time()

    communicating_grad_times.append((end_communicating_grad_time - start_communicating_grad_time) * 1000)
    total_times.append((end_time - start_time) * 1000)

    print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {summarize_tensor(next(model.parameters()))}", flush=True)
  
  total_time_per_training_step = np.mean(total_times)
  communicating_grad_time_per_training_step = np.mean(communicating_grad_times)
  print(f"[data_parallelism] Rank {rank}, total_time_per_training_step: {total_time_per_training_step:.2f} ms, communicating_grad_time_per_training_step: {communicating_grad_time_per_training_step:.2f}ms, protation: {(communicating_grad_time_per_training_step / total_time_per_training_step) * 100}%")
  cleanup()


def benchmark_main(
  num_warmups: int,
  num_trials: int,
  model_size: str,
):
  if model_size == "small":
    model_configuration = {
      "d_model": 768,
      "d_ff": 3072,
      "num_layers": 12,
      "num_heads": 12
    }
  elif model_size == "medium":
    model_configuration = {
      "d_model": 1024,
      "d_ff": 4096,
      "num_layers": 24,
      "num_heads": 24
    }
  elif model_size == "large":
    model_configuration = {
      "d_model": 1280,
      "d_ff": 5120,
      "num_layers": 36,
      "num_heads": 20
    }
  elif model_size == "xl":
    model_configuration = {
      "d_model": 1600,
      "d_ff": 6400,
      "num_layers": 48,
      "num_heads": 35
    }
  elif model_size == "2.7B":
    model_configuration = {
      "d_model": 2560,
      "d_ff": 10240,
      "num_layers": 32,
      "num_heads": 32
    }
  model_configuration["vocab_size"] = 10000
  model_configuration["context_length"] = 256
  model_configuration["rope_theta"] = 10000

  batch_size = 4
  batch_x = torch.randint(
    low=0, 
    high=model_configuration["vocab_size"], 
    size=(batch_size, model_configuration["context_length"]),
    dtype=torch.long
  )

  batch_y = torch.randint(
    low=0, 
    high=model_configuration["vocab_size"], 
    size=(batch_size, model_configuration["context_length"]),
    dtype=torch.long
  )

  print(f'batch_x.shape: {batch_x.shape}')
  print(f'batch_y.shape: {batch_y.shape}')


  spawn(
    data_parallelism_main_with_benchmarking, 
    world_size=2, 
    batch_x=batch_x, 
    batch_y=batch_y, 
    num_warmups=num_warmups, 
    num_trials=num_trials, 
    model_configuration=model_configuration
  )

if __name__ == "__main__":
  benchmark_main(
    num_warmups=5,
    num_trials=10,
    model_size="large"
  )
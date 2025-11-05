import os
import sys
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from inspect import isfunction
from typing import Callable
from cs336_systems.parallelism.utils.torch_util import get_device

class DisableDistributed:
  """
  上下文管理器: 临时禁用所有分布式函数, 将其替换为无操作的空函数。
  使用场景：
    - 调试时避免启动多进程
    - 在单进程环境下测试分布式代码逻辑
    - 当调试器正在跟踪时 (sys.gettrace()返回非空)
  """
  def __enter__(self):
    self.old_functions = {}
    for name in dir(dist):
      value = getattr(dist, name, None)
      if isfunction(value):
        self.old_functions[name] = value
        setattr(dist, name, lambda *args, **kwargs: None)

  def __exit__(self, exc_type, exc_value, traceback):
    for name in self.old_functions:
      setattr(dist, name, self.old_functions[name])

def spawn(func: Callable, world_size: int, *args, **kwargs):
  """
  使用示例:
    def train_fn(rank, world_size, learning_rate, batch_size):
      setup(rank, world_size) # 初始化分布式
      # ... 训练逻辑
    
    # 使用spawn调用
    spawn(train_fn, wold_size=4, learning_rate=0.01, batch_size=32)
  """
  # 注意：这里假设 kwargs 的顺序与 func函数需要的参数顺序一致
  if sys.gettrace():  # 检查是否在调试模式下
    # 调试模式：直接运行函数（单进程）
    with DisableDistributed():
      args = (0, world_size,) + args + tuple(kwargs.values())
      func(*args) # 以 rank=0 单进程运行
  else:
    # 正常模式：使用多进程启动
    args = (world_size,) + args + tuple(kwargs.values())
    # 使用 torch.multiprocessing.spawn 启动多进程，每个进程运行指定的函数
    mp.spawn(func, args=args, nprocs=world_size, join=True)

def setup(rank: int, world_size: int):
  """
  初始化设置函数，用于在多机多卡或单机多卡训练时初始化各个进程，使它们能够相互通信。
  其中, 主节点 (rank 0) 负责协调工作, 实际的数据传输则通过后端 (NCCL或Gloo) 进行。
  Args:
    rank (int): 当前进程的id (0 表示主节点)
    world_size (int): 参与训练的总进程数
  Note:
    需要在每个训练进程中调用此函数初始化进程
  """
  # 设置主节点地址和端口
  os.environ["MASTER_ADDR"] = "localhost"   # 制定主节点（rank 0）的主机地址，这里使用本地主机
  os.environ["MASTER_PORT"] = "15623"       # 设置通信端口号

  # 初始化进程组
  if torch.cuda.is_available():
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
  else:
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

def cleanup():
  torch.distributed.destroy_process_group()

def render_duration(duration: float) -> str:
  if duration < 1e-3:
    return f"{duration * 1e6:.2f}us"
  if duration < 1:
    return f"{duration * 1e3:.2f}ms"
  return f"{duration:.2f}s"

def int_divide(a: int, b: int):
  """
  Return a / b and throw an error if there's a remainder.
  """
  assert a % b == 0
  return a // b

def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
  torch.random.manual_seed(0) # For reproducibility
  return nn.Parameter(torch.randn(num_inputs, num_outputs, device=get_device(rank)) / math.sqrt(num_outputs))

def summarize_tensor(tensor: torch.Tensor) -> str:
  return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"
import math
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.optim import Optimizer
from typing import Type, Any

class DDPOptimizer(Optimizer):
  def __init__(self, params: list[nn.Parameter], optimizer_cls: Type[Optimizer], **kwargs: Any):
    self.full_params = list(params)

    # 元信息
    self.rank = dist.get_rank()
    self.world_size = dist.get_world_size()

    # 1. 将参数划分给不同的GPU
    # 简单轮询的方式进行划分，每个rank只管理和更新索引为 rank, rank + world_size, rank + 2*world_size, ... 的参数
    self.local_params = self.full_params[self.rank::self.world_size]

    # 2. 每个GPU对自己负责的参数实例化Optimizer
    # 从而每个GPU只存储了自己负责的参数部分的优化器状态，完成优化器状态分片
    self.optimizer = optimizer_cls(params=self.local_params, **kwargs)

    # 调用父类的构造函数，并传递至完整的参数列表，这样Pytorch的Optimizer框架才能正确识别所有参数
    super().__init__(params=self.full_params, defaults={})

  def step(self, closure=None, **kwargs):
    # 1. 每个GPU先对自己负责的参数进行优化
    loss = self.optimizer.step(closure, **kwargs)

    # 2. 对优化之后的参数进行同步
    # 每个rank更新完自己的local_params之后，需要将这些新值同步给所有其他rank
    # 遍历所有参数，由该参数的“所有者/更新者”rank将其广播出去
    for i, param in enumerate(self.full_params):
      # 计算哪一个rank是该参数的所有者
      owner_rank = i % self.world_size
      # 由所有者rank将更新后的参数值广播给所有进程，注意，这里所有的进程都会同时执行下面的广播语句，这样才可以正确广播
      dist.broadcast(param.data, src=owner_rank)

    return loss

  def zero_grad(self, set_to_none = True):
    return self.optimizer.zero_grad(set_to_none=set_to_none)

  def add_param_group(self, param_group: dict[str, Any]):
    # 将该参数组添加到父类中，以进行跟踪
    super().add_param_group(param_group)
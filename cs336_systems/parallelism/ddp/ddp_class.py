import torch
import torch.nn as nn
import torch.distributed as dist

class DDP(nn.Module):
  """
  Example:
    dist.init_process_group(...)
    model = DDP(MyModel())
    for batch in dataloader:
      output = model(batch) # 前向传播
      loss = criterion(output, target)

      optimizer.zero_grad(set_to_none=True)
      loss.backward() # 反向传播 (触发梯度钩子)

      model.finish_gradient_synchronization() # 等待梯度同步完成
      optimizer.step()  # 更新参数
      optimizer.zero_grad()
  """
  def __init__(self, module: nn.Module):
    super().__init__()

    self.module = module
    self.handlers = []  # 存储异步操作句柄

    if not dist.is_initialized():
      raise RuntimeError("torch.distributed is not initialized.")

    # 1.参数广播（确保所有GPU上的参数保持一致性）
    self._broadcast_model()

    # 2.为每个参数注册一个在grad累积之后运行的反向钩子
    self._register_grad_hooks()
  
  def _broadcast_model(self):
    """
    广播模型参数, 使初始化时所有GPU上的参数保持一致性.
    实现细节:
      1.遍历模型的所有参数, 对每个参数都调用dist.broadcast(param.data, src=0)
    注意:
      - 广播参数时广播的是参数的权重值param.data, 而非param本身;
      - src=0 确保从主进程开始广播;
      - 目前只能用这种对每个参数单独调用broadcast对方式来广播参数, 实现过batch_braodcast_model:
        使用torch.nn.utils.parameters_to_vector和torch.nn.utils.vector_to_parameters分别先展平, 然后广播, 最后还原的操作, 但是失败了, 目前还不知道原因. 
    """
    for param in self.module.parameters():
      dist.broadcast(param.data, src=0)
  
  def _register_grad_hooks(self):
    for param in self.module.parameters():
      if param.requires_grad:
        param.register_post_accumulate_grad_hook(self._transform_grad)
  
  def _transform_grad(self, param):
    """
    在grad累积之后运行的反向钩子, 该钩子将会对该参数的梯度使用all_reduce进行梯度同步, 且使用的是异步的方式 (即async_op=True).

    实现细节:
      1.为了兼容性(后端为gloo或者是nccl): 统一使用手动求平均的方法, 首先各个GPU对自己所持有的梯度进行平均 (除以world_size)
      2.使用dist.all_reduce进行梯度同步, 且设置 async_op=True, 其返回一个handler, 我们将其添加到 self.handlers中, 待之后可以使用该handler.wait() 来等待梯度同步完成
    """
    with torch.no_grad():
      param.grad.data /= dist.get_world_size()
    
    self.handlers.append(dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True))


  def forward(self, *inputs, **kwargs):
    return self.module(*inputs, **kwargs)
  
  def finish_gradient_synchronization(self):
    """
    该函数用于等待所有梯度同步完成, 通常在 optimizer.step() 之前调用.
    实现细节:
      1.遍历self.handlers中的所有handler, 并调用handler.wait(), 使得可以等待所有参数的异步的"梯度同步操作"完成
      2.最后要清空self.handlers, 因为下一次反向传播时(即每次反向传播时, 都会重新触发梯度累积之后运行的反向钩子, 从而重新调用dist.all_reduce生成新的handler)
    """
    for handler in self.handlers:
      handler.wait()
    self.handlers.clear()
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.utils import parameters_to_vector, vector_to_parameters

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
  def __init__(self, module: nn.Module, bucket_size_mb: float):
    super().__init__()

    self.module = module
    self.bucket_size_bytes = bucket_size_mb * 1024**2  # 一个参数桶的内存上限

    # 在一次反向传播中存储异步操作句柄和对应的梯度缓冲区
    self.grad_handles = []
    self.flat_grads_list = []

    if not dist.is_initialized():
      raise RuntimeError("torch.distributed is not initialized.")

    self.world_size = dist.get_world_size()

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
    """
    核心逻辑: 将参数分桶, 并为每个同中最后一个计算梯度的参数注册一个钩子.
    这个钩子将在该参数的梯度计算完成后触发, 然后也意味着这整个桶的梯度已经计算完成, 则可以对整个桶的梯度进行 all-reduce.
    """
    params_with_grad = [p for p in self.module.parameters() if p.requires_grad]

    # 反向遍历参数-与梯度计算的顺序一致 (从后向前), 这样可以更好地重叠计算和通信
    bucket = []
    current_bucket_size = 0

    for param in reversed(params_with_grad):
      param_size = param.numel() * param.element_size()
      
      # 如果将当前参数加入桶中会超过大小限制，则处理当前桶
      if current_bucket_size + param_size > self.bucket_size_bytes and len(bucket) > 0:
        self._add_hook_for_bucket(bucket)
        bucket = []
        current_bucket_size = 0
      
      bucket.append(param)
      current_bucket_size += param_size
    
    # 处理最后一个未满的桶
    if len(bucket) > 0:
      self._add_hook_for_bucket(bucket)
  
  def _add_hook_for_bucket(self, bucket: list[nn.Parameter]):
    """
    为一个桶的参数创建并注册钩子.
    钩子会绑定在桶里面的最后一个参数上 (因为我们是反向遍历的, 所以这个参数是桶里面最晚计算出梯度的)
    """
    # 钩子函数将使用这个桶的参数列表
    def hook_factory(bucket_params):
      def _bucket_grad_hook(*args, **kwargs):
        # 1. 将桶内所有参数的梯度拼接成一个扁平的张量
        grads = [p.grad.data for p in bucket_params]
        flat_grad = parameters_to_vector(grads)
        self.flat_grads_list.append(flat_grad)

        # 2. 对梯度求平均
        flat_grad /= self.world_size

        # 3. 启动异步 all-reduce 并保存2句柄
        handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
        self.grad_handles.append((handle, bucket_params))
      return _bucket_grad_hook

    # 钩子注册在桶的最后一个参数上（反向传播时的最后一个参数）
    trigger_param = bucket[-1]
    trigger_param.register_post_accumulate_grad_hook(hook_factory(bucket))

  def forward(self, *inputs, **kwargs):
    return self.module(*inputs, **kwargs)
  
  def finish_gradient_synchronization(self):
    """
    该函数用于等待所有梯度同步完成, 并在完成后将同步好的扁平梯度复制回原参数的.grad属性. 通常在 optimizer.step() 之前调用.
    """
    for handle, bucket_params in self.grad_handles:
      handle.wait()
    
    flat_grad_iter = iter(self.flat_grads_list)
    for _, bucket_params in self.grad_handles:
      flat_grad = next(flat_grad_iter)
      grad_views = [p.grad.data for p in bucket_params]

      offset = 0
      for grad in grad_views:
        numel = grad.numel()
        grad.copy_(flat_grad[offset:offset+numel].view_as(grad))
        offset += numel
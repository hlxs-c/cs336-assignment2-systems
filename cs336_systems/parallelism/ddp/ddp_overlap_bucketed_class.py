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

    if not dist.is_initialized():
      raise RuntimeError("torch.distributed is not initialized.")

    self.world_size = dist.get_world_size()

    # 在一次反向传播中用于梯度同步需要存储的状态
    # 每个元素是一个字典: {'params': [...], 'handle': None, 'flat_grad': None}, 
    # 'params' 字段是参数桶； 'handle'是为这个参数桶里所有参数的梯度展平后执行异步all_reduce的操作句柄; flat_grad 是这个参数桶里所有参数的梯度展平后的向量
    self.bucket_info_list = []

    # 1.参数广播（确保所有GPU上的参数保持一致性）
    self._broadcast_model()

    # 2.为每个参数注册一个在grad累积之后运行的反向钩子
    self._prepare_buckets()

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

  def _prepare_buckets(self):
    """
    遍历参数进行分桶, 并为每个桶注册钩子.
    """
    params_with_grad = [p for p in self.module.parameters() if p.requires_grad]
    
    bucket = []
    current_bucket_size = 0

    for param in reversed(params_with_grad):
      param_size = param.numel() * param.element_size()

      if current_bucket_size + param_size > self.bucket_size_bytes and len(bucket) > 0:
        self._register_hook_for_bucket(bucket)
        bucket = []
        current_bucket_size = 0
      
      bucket.append(param)
      current_bucket_size += param_size

    if len(bucket) > 0:
      self._register_hook_for_bucket(bucket)
  
  def _register_hook_for_bucket(self, bucket: list[nn.Parameter]):
    """
    为桶创建结构化信息, 并为触发参数注册钩子.
    """
    # 存储桶的参数列表
    bucket_info = {'params': bucket}
    self.bucket_info_list.append(bucket_info)

    def hook_factory(info):
      def _bucket_grad_hook(*args, **kwargs):
        # 1. 将桶内所有参数的梯度拼接展平（flatten）为一个扁平的张量
        grads = []
        for p in info['params']:
          if p.grad is None:
            # 如果参数没有梯度，创建一个零张量替代
            grads.append(torch.zeros_like(p.data))
          else:
            grads.append(p.grad.data)
        flat_grad = parameters_to_vector(grads)

        # 2. 将创建的扁平梯度保存到桶的信息字典中
        info['flat_grad'] = flat_grad

        # 3. 对梯度求平均
        flat_grad /= self.world_size

        # 4. 启动异步 all-reduce 并保存句柄
        handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
        info['handle'] = handle
      return _bucket_grad_hook

    # 钩子注册在桶的最后一个参数上（反向传播时最晚计算梯度的参数）
    trigger_param = bucket[-1]
    trigger_param.register_post_accumulate_grad_hook(hook_factory(bucket_info))

  def forward(self, *inputs, **kwargs):
    return self.module(*inputs, **kwargs)
  
  def finish_gradient_synchronization(self):
    """
    该函数用于等待所有梯度同步完成, 并在完成后将同步好的扁平梯度复制回原参数的.grad属性. 通常在 optimizer.step() 之前调用.
    """
    for bucket_info in self.bucket_info_list:
      # 确保句柄已经被创建（即钩子函数已经在反向传播过程中被触发）
      if bucket_info['handle'] is not None:
        bucket_info['handle'].wait()

        # 获取同步完成的扁平梯度
        synced_flat_grad = bucket_info['flat_grad']

        # 获取桶内参数对应的 .grad 属性列表
        # 注意：这里传递的是参数的 .grad 属性本身，而不是 .grad.data，因为之后的 vector_to_parameters 会直接修改这些张量的内容
        grads_to_update = [p.grad for p in bucket_info['params']]

        # 将扁平梯度的数据复制回各个参数的 .grad 字段
        vector_to_parameters(synced_flat_grad, grads_to_update)

        # 重置当前迭代留下的状态（句柄和梯度缓冲区），为下一轮做准备
        # 注意：不能直接清空self.bucket_info_list，因为分桶只会做一次，所以bucket_info_list也只会把每个桶的状态结构构建一次，之后都是直接修改
        bucket_info['handle'] = None
        bucket_info['flat_grad'] = None
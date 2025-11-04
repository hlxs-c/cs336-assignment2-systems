import math
import torch

from einops import rearrange

def _single_batch_flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
  """
  Args:
    Q (torch.Tensor): shape [seq_len_q, d] or [N_q, d]
    K (torch.Tensor): shape [seq_len_k, d] or [N_k, d]
    V (torch.Tensor): shape [seq_len_k, d] or [N_k, d]
  """
  # 1. 获取输入数据元信息
  N_q, d = Q.shape
  N_k, d_k = K.shape
  
  assert d == d_k, "Q和K的嵌入维度必须相同"
  
  # 2. 初始化输出矩阵O 和 Log-sum-exp
  O = torch.zeros(N_q, d, device=Q.device, dtype=Q.dtype)
  L = torch.zeros(N_q, device=Q.device, dtype=Q.dtype)

  # 3. 分块配置
  B_q = 32  # 查询块大小
  B_k = 128 # 键值块大小 （通常比B_q大，以利用GPU并行性）

  T_q = math.ceil(N_q / B_q)  # 查询块数量
  T_k = math.ceil(N_k / B_k)  # 键值块数量

  # 4. 主循环
  for i in range(T_q):
    # 加载查询块 Q_i
    start_q = i * B_q
    end_q = min((i + 1) * B_q, N_q)
    Q_i = Q[start_q:end_q]  # [actual_B_q, d]
    actual_B_q = Q_i.shape[0]

    # 初始化当前查询块的中间结果
    O_i = torch.zeros(actual_B_q, d, device=Q.device, dtype=Q.dtype)
    m_i = torch.full((actual_B_q,), -torch.inf, device=Q.device, dtype=Q.dtype) # 最大值，初始化为负无穷
    l_i = torch.zeros(actual_B_q, device=Q.device, dtype=Q.dtype) # 累加和

    for j in range(T_k):
      # 加载键值块
      start_k = j * B_k
      end_k = min((j + 1) * B_k, N_k)
      K_j = K[start_k:end_k] # [actual_B_k, d]
      V_j = V[start_k:end_k] # [actual_B_k, d]
      actual_B_k = K_j.shape[0]

      # 计算注意力分数 S_ij = Q_i @ K_j^T
      S_ij = Q_i @ K_j.T  # [actual_B_q, actual_B_k]
      S_ij *= d**-0.5 # 缩放因子

      # 应用因果掩码（如果启用）
      if is_causal:
        # 创建向量化的因果掩码
        row_indices = torch.arange(start_q, end_q, device=Q.device).unsqueeze(1)  # [actual_B_q, 1]
        col_indices = torch.arange(start_k, end_k, device=Q.device).unsqueeze(0)  # [1, actual_B_k]
        causal_mask = row_indices < col_indices # 广播机制 causal_mask 为True的位置代表需要被掩码的
        S_ij = S_ij.masked_fill(causal_mask, -torch.inf)

      # 计算当前块的最大值
      m_ij = torch.max(S_ij, dim=1).values  # [actual_B_q]

      # 计算新的全局最大值
      m_new = torch.maximum(m_i, m_ij)

      # 计算当前块的指数（数值稳定性版本，先对S_ij 使用最大值进行缩放，再求指数）
      P_ij = torch.exp(S_ij - m_new.unsqueeze(1)) # [actual_B_q, actual_B_k]
      # 计算当前块的注意力加权值
      P_ij_V_j = P_ij @ V_j # [actual_B_q, d]

      # 计算当前块的每一行的和
      l_ij = torch.sum(P_ij, dim=1) # [actual_B_q]

      # 计算新的累加和（考虑之前最大值的影响）
      l_new = l_i * torch.exp(m_i - m_new) + l_ij

      # 更新输出矩阵 O_i
      # 1. 首先缩放之前的 O_i （由于最大值变化）
      scale_factor = torch.exp(m_i - m_new) # [actual_B_q]
      O_i = scale_factor.unsqueeze(1) * O_i # 对每一行进行缩放

      # 2. 加上当前块的贡献
      O_i = O_i + P_ij_V_j

      # 更新中间状态
      m_i = m_new
      l_i = l_new
    
    # 当前查询块处理完成，进行最终的归一化
    O_i = O_i / l_i.unsqueeze(1)  # 每行除以对应的累加和

    # 将当前查询块结果写入最终输出
    O[start_q:end_q] = O_i

    # 将当前查询块log-sum-exp写入L
    L[start_q:end_q] = m_i + torch.log(l_i)
  
  return O, L



class PytorchFlashAttention2(torch.autograd.Function):
  @staticmethod
  def forward(
    ctx,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False
  ):
    batch_size = Q.shape[0]

    # 处理每个批次
    outputs = []
    log_sum_exps = []
    for i in range(batch_size):
      cur_output, cur_L = _single_batch_flash_attention_forward(Q=Q[i], K=K[i], V=V[i], is_causal=is_causal)
      outputs.append(cur_output)
      log_sum_exps.append(cur_L)
    
    # 堆叠结果
    out = torch.stack(outputs, dim=0)
    L = torch.stack(log_sum_exps, dim=0)

    # 保存用于反向传播
    ctx.save_for_backward(Q, K, V, out, L)
    ctx.is_causal = is_causal

    return out

  @staticmethod
  def backward(ctx, grad_output):
    raise NotImplementedError

def test_single_batch_flash_attention():
  torch.manual_seed(42)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device == "cpu":
    print("cuda is not available, exit.")
    exit()

  # 创建测试数据
  N_q, N_k, d = 100, 200, 64
  Q = torch.randn(N_q, d, device=device)
  K = torch.randn(N_k, d, device=device)
  V = torch.randn(N_k, d, device=device)

  # 测试非因果注意力
  print("测试非因果注意力...")
  O_flash_attention, _ = _single_batch_flash_attention_forward(Q, K, V, is_causal=False) 

  # 与标准注意力比较
  S_standard = Q @ K.T
  S_standard *= d**-0.5
  P_standard = torch.softmax(S_standard, dim=-1)
  O_standard = P_standard @ V

  error = torch.abs(O_flash_attention - O_standard).max()
  print(f"最大误差: {error.item()}")
  print(f"结果形状: {O_flash_attention.shape}")
  print(f'torch.allclose: {torch.allclose(O_flash_attention, O_standard, rtol=1e-1, atol=1e-2)}')

  # 测试因果注意力
  print("\n测试因果注意力...")
  O_causal, _ = _single_batch_flash_attention_forward(Q, K, V, is_causal=True)

  # 创建因果掩码的标准实现
  mask = torch.tril(torch.ones(N_q, N_k, device=device))
  S_causal_standard = Q @ K.T
  S_causal_standard *= d**-0.5
  S_causal_standard = S_causal_standard.masked_fill(mask == 0, -torch.inf)
  P_causal_standard = torch.softmax(S_causal_standard, dim=-1)
  O_causal_standard = P_causal_standard @ V
  
  causal_error = torch.abs(O_causal - O_causal_standard).max()
  print(f"因果注意力最大误差: {causal_error.item()}")
  print(f'torch.allclose: {torch.allclose(O_flash_attention, O_standard, rtol=1e-1, atol=1e-2)}')

if __name__ == "__main__":
  test_single_batch_flash_attention()
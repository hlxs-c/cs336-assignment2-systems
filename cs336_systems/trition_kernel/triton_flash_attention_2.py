import torch
import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
  Q_ptr, K_ptr, V_ptr,
  O_ptr, L_ptr,
  stride_qb, stride_qq, stride_qd,
  stride_kb, stride_kk, stride_kd,
  stride_vb, stride_vk, stride_vd,
  stride_ob, stride_oq, stride_od,
  stride_lb, stride_lq,
  N_QUERIES, N_KEYS,
  scale,
  D: tl.constexpr,
  Q_TILE_SIZE: tl.constexpr,
  K_TILE_SIZE: tl.constexpr,
  is_causal: tl.constexpr
):
  # Program indices
  query_tile_index = tl.program_id(0)
  batch_index = tl.program_id(1)

  # Offset each pointer with correspoding batch index
  # multiplied with the batch stride for each tensor
  Q_block_ptr = tl.make_block_ptr(
    base=Q_ptr + batch_index * stride_qb, # 从该实例对应的查询Q开始（batch_index * stride_qb 代表从Q_ptr 到当前这个Q的位移）
    shape=(N_QUERIES, D),
    strides=(stride_qq, stride_qd),
    offsets=(query_tile_index * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0)
  )

  K_block_ptr = tl.make_block_ptr(
    base=K_ptr + batch_index * stride_kb,
    shape=(N_KEYS, D),
    strides=(stride_kk, stride_kd),
    offsets=(0, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(1, 0)
  )

  V_block_ptr = tl.make_block_ptr(
    base=V_ptr + batch_index * stride_vb,
    shape=(N_KEYS, D),
    strides=(stride_vk, stride_vd),
    offsets=(0, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(1, 0)
  )

  O_block_ptr = tl.make_block_ptr(
    base=O_ptr + batch_index * stride_ob,
    shape=(N_QUERIES, D),
    strides=(stride_oq, stride_od),
    offsets=(query_tile_index * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0)
  )

  L_block_ptr = tl.make_block_ptr(
    base=L_ptr + batch_index * stride_lb,
    shape=(N_QUERIES,),
    strides=(stride_lq,),
    offsets=(query_tile_index * Q_TILE_SIZE,),
    block_shape=(Q_TILE_SIZE,),
    order=(0,)
  )

  # 加载数据 Q_i
  Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

  # 初始化输出以及中间变量
  O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
  m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
  l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

  for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
    # 1. 加载数据K_j, V_j
    K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # 2.计算注意力分数 并使用 scale 进行缩放（缩放点击）
    S_ij = tl.dot(Q_i, K_j.T) * scale   # [B_q, B_k]

    if is_causal:
      # 获取当前查询块和键块的起始位置
      q_start_index = query_tile_index * Q_TILE_SIZE
      k_start_index = j * K_TILE_SIZE

      # 创建一个范围张量，表示块内的偏移量
      q_offsets = tl.arange(0, Q_TILE_SIZE)
      k_offsets = tl.arange(0, K_TILE_SIZE)

      # 计算每个元素的绝对位置
      # q_pos 的 shape 为 [Q_TILE_SIZE, 1]
      # k_pos 的 shape 为 [1, K_TILE_SIZE]
      q_pos = q_start_index + q_offsets
      k_pos = k_start_index + k_offsets
      
      # 创建掩码：当 key 的位置 > query 的位置时，为 True (需要被掩盖)
      # 通过广播机制，mask 的 shape 为 [Q_TILE_SIZE, K_TILE_SIZE]
      mask = q_pos[:, None] < k_pos[None, :]

      # 应用掩码，将未来位置的分数设置为负无穷
      S_ij = tl.where(mask, -float('inf'), S_ij)

    # 3.计算当前注意力分数块每一行的最大值
    m_ij = tl.max(S_ij, axis=1) # [B_q]

    # 4. 计算新的全局最大值
    m_new = tl.maximum(m_i, m_ij)

    # 5. 计算当前注意力分数块的指数（数值稳定性版本）
    P_ij = tl.exp(S_ij - m_new[:, None])  # [B_q, B_k]

    # 6. 使用当前注意力分数块与V_j 计算注意力加权值
    P_ij_V_j = tl.dot(P_ij, V_j)  # [B_q, d]

    # 7. 计算当前注意力分数块的每一行的指数和
    l_ij = tl.sum(P_ij, axis=1) # [B_q,]

    # 8. 计算缩放因子
    scale_factor = tl.exp(m_i - m_new)

    # 9. 计算新的全局每一行的指数和（注意最大值变化的影响）
    l_new = scale_factor * l_i + l_ij

    # 9. 更新输出 O_i （注意最大值变化的影响，需要对之前的进行缩放）
    O_i = scale_factor[:, None] * O_i + P_ij_V_j

    # 10. 更新中间状态
    m_i = m_new
    l_i = l_new

    # 11. 移动指针到下一个块
    K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
    V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
  
  # 循环结束后，对 O_i 进行归一化
  O_i /= l_i[:, None]

  # 计算 log-sum-exp
  L_i = m_i + tl.log(l_i)

  # 将结果和log-sum-exp保存
  tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
  tl.store(L_block_ptr, L_i, boundary_check=(0,))

class TritonFlashAttention2(torch.autograd.Function):
  @staticmethod
  def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool):
    batch_size, seq_len_q, d = Q.shape
    batch_size, seq_len_k, d = K.shape

    N_queries = seq_len_q
    N_keys = seq_len_k
    Q_TILE_SIZE = 32  # B_q
    K_TILE_SIZE = 32   # B_k

    T_q = triton.cdiv(N_queries, Q_TILE_SIZE)
    T_k = triton.cdiv(N_keys, K_TILE_SIZE)

    O = torch.zeros(batch_size, seq_len_q, d, device=Q.device, dtype=Q.dtype)
    L = torch.zeros(batch_size, seq_len_q, device=Q.device, dtype=Q.dtype)

    grid = (T_q, batch_size)
    
    flash_fwd_kernel[grid](
      Q_ptr=Q, K_ptr=K, V_ptr=V,
      O_ptr=O, L_ptr=L,
      stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
      stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
      stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
      stride_ob=O.stride(0), stride_oq=O.stride(1), stride_od=O.stride(2),
      stride_lb=L.stride(0), stride_lq=L.stride(1),
      N_QUERIES=N_queries, N_KEYS=N_keys,
      scale=d**-0.5,
      D=d,
      Q_TILE_SIZE=Q_TILE_SIZE,
      K_TILE_SIZE=K_TILE_SIZE,
      is_causal=is_causal
    )

    ctx.save_for_backward(Q, K, V, O, L)

    return O

  @staticmethod
  def backward(ctx, *grad_outputs):
    return super().backward(ctx, *grad_outputs)


if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available else "cpu"

  if device == "cpu":
    print("CUDA not available, skipping test.")
    exit()

  batch_size = 32
  seq_len = 256
  d_model = 128
  Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
  K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
  V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

  print(f'Q.shape: {Q.shape}')
  print(f'K.shape: {K.shape}')
  print(f'V.shape: {V.shape}')

  # Triton pass
  output_triton = TritonFlashAttention2.apply(Q, K, V, False)

  S_standard = Q @ K.transpose(1, 2) * d_model**-0.5
  P_standard = torch.softmax(S_standard, dim=-1)
  O_standard = P_standard @ V

  print(f'torch.allclose: {torch.allclose(output_triton, O_standard, rtol=1e-1, atol=1e-3)}')
import torch
import triton
import triton.language as tl
import numpy as np

from einops import rearrange

def weighted_sum(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
  # 实际上就是 矩阵-向量 乘法，与 x @ weight 是等价的
  return (weight * x).sum(axis=-1)


@triton.jit
def weighted_sum_fwd(
  x_ptr, weight_ptr,  # Input pointers
  output_ptr, # Output pointer
  x_stride_row, x_stride_dim, # Strides tell us how to move one element in each axis of a tensor
  weight_stride_dim,
  output_stride_row,
  ROWS, D,
  ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,  # Tile shapes must be known at compile time
):
  """
  内核代码，运行在GPU上，由成百上千个线程并行执行，完成实际的数值计算。
  """
  # 1. 确定当前实例的身份和任务
  row_tile_idx = tl.program_id(0)

  # 2. 创建“智能指针“（Block Pointers）
  """
  tl.make_block_ptr 创建“智能指针”，它需要知道：
  - base: 整个张量（如 x_2d）的起始内存地址
  - shape: 整个张量的形状 （如 (n_rows, D)）
  - strides: 内存布局
  - offsets: 定义这个指针的初始指向位置。例如对于 x_block_ptr, offsets=(row_tile_idx * ROWS_TILE_SIZE, 0) 意味着第 k 个实例的该指针会从第 k * ROWS_TILE_SIZE 行、第 0 列开始
  - block_shape: 定义了调用 tl.load 时一次加载的数据块大小
  """
  x_block_ptr = tl.make_block_ptr(
    base=x_ptr,
    shape=(ROWS, D,),
    strides=(x_stride_row, x_stride_dim),
    offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
    block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
    order=(1, 0),
  )

  weight_block_ptr = tl.make_block_ptr(
    base=weight_ptr,
    shape=(D,),
    strides=(weight_stride_dim,),
    offsets=(0,),
    block_shape=(D_TILE_SIZE,),
    order=(0,),
  )

  output_block_ptr = tl.make_block_ptr(
    base=output_ptr,
    shape=(ROWS,),
    strides=(output_stride_row,),
    offsets=(row_tile_idx * ROWS_TILE_SIZE,),
    block_shape=(ROWS_TILE_SIZE,),
    order=(0,),
  )

  # 3. 初始化累加器：在SRAM中创建一个局部变量，用于存放中间计算结果。每个实例都有自己的 accumulator
  accumulator = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

  # 4. 沿 D 维度分块循环计算：通常，向量 weight 的维度 D 可能很大，无法一次性加载到缓存中。因此，需要一个循环，每次处理 D_TILE_SIZe 这么长的一个小段
  for i in range(tl.cdiv(D, D_TILE_SIZE)):
    # 4.a 从全局内存加载数据块
    
    # 从 x_2d 的当值位置加载一个 （ROWS_TILE_SIZE, D_TILE_SIZE) 大小的数据块到寄存器。
    # boundary_check 和 padding_option 处理边界情况，如果加载到块超出了张量的实际边界，boundary_check 会检测到，并用 0 来填充超出部分，从而避免内存访问错误
    row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  
    weight_chunk = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

    # 4.b 进行数值计算并将结果累加
    accumulator += tl.sum(row * weight_chunk[None, :], axis=1)

    # 4.c 移动指针到下一个 D 维度的块
    x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))   # x_block_ptr 在列方向上移动 D_TILE_SIZE
    weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))   # weight_block_ptr 移动 D_TILE_SIZE

  # 5. 将最终结果写回全局内存：当循环结束后，accumulator 中保存了分配给这个实例的 ROWS_TILE_SIZE 行数据完整的加权和
  tl.store(output_block_ptr, accumulator, boundary_check=(0,))

@triton.jit
def weighted_sum_backward(
  x_ptr, weight_ptr, #  Input
  grad_output_ptr,  # Grad input
  grad_x_ptr, partial_grad_weight_ptr,  # Grad outputs
  stride_xr, stride_xd,
  stride_wd,
  stride_gr,
  stride_gxr, stride_gxd,
  stride_gwb, stride_gwd,
  NUM_ROWS, D,
  ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
  row_tile_idx = tl.program_id(0)
  n_row_tiles = tl.num_programs(0)

  grad_output_block_ptr = tl.make_block_ptr(
    base=grad_output_ptr,
    shape=(NUM_ROWS,),
    strides=(stride_gr,),
    offsets=(row_tile_idx * ROWS_TILE_SIZE,),
    block_shape=(ROWS_TILE_SIZE,),
    order=(0,),
  )
  x_block_ptr = tl.make_block_ptr(
    base=x_ptr,
    shape=(NUM_ROWS, D,),
    strides=(stride_xr, stride_xd),
    offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
    block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
    order=(1, 0)
  )
  weight_block_ptr = tl.make_block_ptr(
    base=weight_ptr,
    shape=(D,),
    strides=(stride_wd,),
    offsets=(0,),
    block_shape=(D_TILE_SIZE,),
    order=(0,)
  )
  grad_x_block_ptr = tl.make_block_ptr(
    base=grad_x_ptr,
    shape=(NUM_ROWS, D,),
    strides=(stride_gxr, stride_gxd),
    offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
    block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
    order=(1, 0)
  )
  partial_grad_weight_block_ptr = tl.make_block_ptr(
    base=partial_grad_weight_ptr,
    shape=(n_row_tiles, D,),
    strides=(stride_gwb, stride_gwd),
    offsets=(row_tile_idx, 0),
    block_shape=(1, D_TILE_SIZE),
    order=(1, 0)
  )

  grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")

  for i in range(tl.cdiv(D, D_TILE_SIZE)):
    weight_chunk = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
    grad_x_chunk = grad_output[:, None] * weight_chunk[None, :]
    tl.store(grad_x_block_ptr, grad_x_chunk, boundary_check=(0, 1))

    row_chunk = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
    
    # <<<<<<<<<<<<<<<<<<<< FIX IS HERE <<<<<<<<<<<<<<<<<<<<
    # `tl.sum` does not support `keepdims`. We do it manually.
    # First, sum along axis 0. The result is a 1D tensor of shape (D_TILE_SIZE,).
    grad_weight_partial = tl.sum(row_chunk * grad_output[:, None], axis=0)
    # Then, add a new dimension to make its shape (1, D_TILE_SIZE) for storing.
    grad_weight_chunk = grad_weight_partial[None, :]
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    tl.store(partial_grad_weight_block_ptr, grad_weight_chunk, boundary_check=(0, 1))

    x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))
    weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))
    partial_grad_weight_block_ptr = tl.advance(partial_grad_weight_block_ptr, (0, D_TILE_SIZE))
    grad_x_block_ptr = tl.advance(grad_x_block_ptr, (0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    宿主代码：运行在CPU上，负责准备数据、设定并行计算的配置（网格grid），以及启动Triton内核
    """
    # 1. 获取张量元信息，记录张量 x 的原始形状 input_shape 和 最后一个维度的大小 D
    D = x.shape[-1]
    input_shape = x.shape
    
    # 2. 数据预处理：将输入 x 展平为 2D，把复杂的多维问题简化为一个标准的矩阵-向量乘法问题
    x_2d = x.view(-1, D)
    n_rows = x_2d.shape[0]

    # 保存张量以备反向传播时使用
    ctx.save_for_backward(x_2d, weight)
    
    assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
    assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"

    # 3. 设置内核超参数
    ctx.D_TILE_SIZE = 128
    ctx.ROWS_TILE_SIZE = 32
    ctx.input_shape = input_shape

    # 4. 创建 1D 输出张量，其长度 n_rows 等于展平后的 x_2d 的行数。这个 y 将被传递给 GPU 内核，用于接收计算结果。
    y = torch.zeros(n_rows, device=x.device, dtype=x.dtype)

    # 5. 定义执行网格（Grid）：定义了要启动多少个并行的“程序实例”（在CUDA中称为线程块）
    # 有 n_rows 行数据需要处理，每个程序实例处理 ROWS_TILE_SIZE 行，
    # triton.cdiv(a, b) 是向上取整的除法，确保即使总行数不是 ROWS_TILE_SIZE 的整数倍，也能够启动足够多的程序实例来覆盖所有行
    # grid 是一个一维元组（num_programs, )，表示我们将在一个一维的网格上启动 num_programs 个实例
    grid = (triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)

    # 6. 启动 Triton 内核
    weighted_sum_fwd[grid](
      x_ptr=x_2d,
      weight_ptr=weight,
      output_ptr=y,
      x_stride_row=x_2d.stride(0),  # 
      x_stride_dim=x_2d.stride(1),
      weight_stride_dim=weight.stride(0),
      output_stride_row=y.stride(0),
      ROWS=n_rows,
      D=D,
      ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
      D_TILE_SIZE=ctx.D_TILE_SIZE,
    )

    # 7. 恢复输出形状：内核计算完成后，y 中存储了一维的加权和结果。使用 view 将其恢复为原始输入 x 去掉最后一个维度后的形状
    return y.view(input_shape[:-1])

  @staticmethod
  def backward(ctx, grad_out):
    x, weight = ctx.saved_tensors
    ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
    input_shape = ctx.input_shape
    n_rows, D = x.shape

    grad_out_1d = grad_out.reshape(-1).contiguous()
    
    n_row_tiles = triton.cdiv(n_rows, ROWS_TILE_SIZE)
    partial_grad_weight = torch.zeros((n_row_tiles, D), device=x.device, dtype=x.dtype)
    grad_x = torch.empty_like(x)

    grid = (n_row_tiles,)
    weighted_sum_backward[grid](
      x_ptr=x,
      weight_ptr=weight,
      grad_output_ptr=grad_out_1d,
      grad_x_ptr=grad_x,
      partial_grad_weight_ptr=partial_grad_weight,
      stride_xr=x.stride(0),
      stride_xd=x.stride(1),
      stride_wd=weight.stride(0),
      stride_gr=grad_out_1d.stride(0),
      stride_gxr=grad_x.stride(0),
      stride_gxd=grad_x.stride(1),
      stride_gwb=partial_grad_weight.stride(0),
      stride_gwd=partial_grad_weight.stride(1),
      NUM_ROWS=n_rows,
      D=D,
      ROWS_TILE_SIZE=ROWS_TILE_SIZE,
      D_TILE_SIZE=D_TILE_SIZE,
    )
    grad_weight = partial_grad_weight.sum(axis=0)

    # Note: returning gradients directly, no need to store on ctx for standard autograd
    return grad_x.view(input_shape), grad_weight

if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"

  if device == "cpu":
    print("CUDA not available, skipping test.")
    exit()
  
  x_shape = (10, 20, 150)
  x = torch.randn(x_shape, requires_grad=True, device=device)
  weight = torch.randn(x_shape[-1], requires_grad=True, device=device)

  print(f'x.shape: {x.shape}')
  print(f'weight.shape: {weight.shape}')

  # Triton implementation - test forward and backward together
  # This is the standard way to test a custom autograd function
  try:
    torch.autograd.gradcheck(WeightedSumFunc.apply, (x, weight), fast_mode=True)
    print("Gradcheck passed!")
  except Exception as e:
    print(f"Gradcheck failed: {e}")

  # For comparing values more directly
  # Triton Pass
  y_triton = WeightedSumFunc.apply(x, weight)
  grad_output = torch.randn_like(y_triton)
  y_triton.backward(grad_output.clone())
  x_grad_triton = x.grad.clone()
  weight_grad_triton = weight.grad.clone()

  # PyTorch Pass
  x.grad.zero_()
  weight.grad.zero_()
  y_torch = weighted_sum(x, weight)
  y_torch.backward(grad_output.clone())
  x_grad_torch = x.grad.clone()
  weight_grad_torch = weight.grad.clone()

  print(f"\nForward pass allclose: {torch.allclose(y_torch, y_triton, atol=1e-2, rtol=1e-3)}")
  print(f"Backward pass grad_x allclose: {torch.allclose(x_grad_torch, x_grad_triton, atol=1e-2, rtol=1e-3)}")
  print(f"Backward pass grad_weight allclose: {torch.allclose(weight_grad_torch, weight_grad_triton, atol=1e-2, rtol=1e-3)}")
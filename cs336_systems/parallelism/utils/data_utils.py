import torch

def generate_sample_data(batch_size: int = 128, num_dim: int = 1024):
  data = torch.randn(batch_size, num_dim)
  return data
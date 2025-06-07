import torch
import torch.nn.functional as F
import os, torch

torch_path = os.path.dirname(torch.__file__)   # thường là .../Lib/site-packages/torch
print("Torch đang nằm ở:", torch_path)
print("Nội dung bên trong torch/:", os.listdir(torch_path))

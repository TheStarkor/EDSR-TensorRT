from torchsummary import summary
import numpy as np
import torch

from edsr import edsr

model = edsr().cuda().eval()
batch_size = 1

x = torch.randn(batch_size, 3, 240, 240, requires_grad=True).cuda()
torch_out = model(x)

torch.onnx.export(model, x, "edsr.onnx", export_params=True, input_names=['LR'], output_names=['PRED'])

# summary(model, (3, 120, 120))

input = torch.from_numpy(np.ones([3, 240, 240]))
input = input.type(torch.float)
print(input.shape)
input = input.view([-1, 3, 240, 240]).cuda()
print(input.shape)

pred = model(input)
print(pred)
print(pred.shape)
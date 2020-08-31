from torchsummary import summary
import numpy as np
import torch
import time

from edsr import edsr
from params import BATCH_SIZE, IMAGE_SIZE

model = edsr().cuda().eval()

x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, requires_grad=True, dtype=torch.float32).cuda()
torch_out = model(x)

torch.onnx.export(model, x, "edsr.onnx", export_params=True, input_names=['LR'], output_names=['PRED'])

input = torch.from_numpy(np.ones([3, IMAGE_SIZE, IMAGE_SIZE]))
input = input.type(torch.float)
input = input.view([-1, 3, IMAGE_SIZE, IMAGE_SIZE]).cuda()


total_time = 0
for _ in range(1000):
    t1 = time.time()
    pred = model(input)
    t2 = time.time()
    total_time += t2 - t1
    print("cost time: ", t2 - t1)
print("total time: {}, avg time: {}".format(total_time, total_time/1000))
from torchsummary import summary
import numpy as np
import torch
import time

from edsr import edsr
from common import BATCH_SIZE, IMAGE_SIZE, ITERATION, plot

model = edsr().cuda().eval()

with torch.no_grad():
    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, requires_grad=True, dtype=torch.float32).cuda()
    torch_out = model(x)

    torch.onnx.export(model, x, "edsr.onnx", export_params=True, input_names=['LR'], output_names=['PRED'])

    input = torch.from_numpy(np.ones([3, IMAGE_SIZE, IMAGE_SIZE]))
    input = input.type(torch.float)
    input = input.view([-1, 3, IMAGE_SIZE, IMAGE_SIZE]).cuda()

    it, times = [], []

    total_time = 0
    for i in range(ITERATION):
        t1 = time.time()
        pred = model(input)
        t2 = time.time()
        total_time += t2 - t1
        it.append(i)
        times.append(t2 - t1)

    plot(it, times, 'torch.png', 'Pytorch {} inference avg: {0:.4f}'.format(IMAGE_SIZE, total_time/ITERATION), 'Iteration', 'time', ['pytorch'])
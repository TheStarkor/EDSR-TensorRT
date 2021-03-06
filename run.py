import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

from rt import allocate_buffers
from common import IMAGE_SIZE, ITERATION, plot

tensorrt_file_name = "edsr.plan"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

with open(tensorrt_file_name, 'rb') as f:
    engine_data = f.read()
engine = trt_runtime.deserialize_cuda_engine(engine_data)

inputs, outputs, bindings, stream = allocate_buffers(engine)
context = engine.create_execution_context()

input_idx = np.ones([3, IMAGE_SIZE, IMAGE_SIZE])
numpy_input = np.asarray(input_idx).astype(trt.nptype(trt.int32)).ravel()

np.copyto(inputs[0].host, numpy_input)

def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()

    return [out.host for out in outputs]

it, times = [], []

total_time = 0
for i in range(ITERATION):
    t1 = time.time()
    trt_outputs = do_inference(
                        context=context,
                        bindings=bindings,
                        inputs=inputs,
                        outputs=outputs,
                        stream=stream)
    t2 = time.time()
    total_time += t2 - t1
    it.append(i)
    times.append(t2 - t1)

plot(it, times, 'tensorrt.png', 'TensorRT {} inference avg: {0:.4f}'.format(IMAGE_SIZE, total_time/ITERATION), 'Iteration', 'time', ['tensorrt'])
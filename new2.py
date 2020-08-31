import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

tensorrt_file_name = "edsr.plan"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

with open(tensorrt_file_name, 'rb') as f:
    engine_data = f.read()
engine = trt_runtime.deserialize_cuda_engine(engine_data)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

inputs, outputs, bindings, stream = [], [], [], []

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))

    if engine.binding_is_input(binding):
        inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
        outputs.append(HostDeviceMem(host_mem, device_mem))

context = engine.create_execution_context()

input_idx = np.ones([3, 240, 240])
numpy_input = np.asarray(input_idx).astype(trt.nptype(trt.int32)).ravel()

print(numpy_input.shape)

np.copyto(inputs[0].host, numpy_input)

def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()

    return [out.host for out in outputs]

 
import time

for _ in range(1):
    t1 = time.time()

    trt_outputs = do_inference(
                        context=context,
                        bindings=bindings,
                        inputs=inputs,
                        outputs=outputs,
                        stream=cuda.Stream())
    trt_outputs = np.array(trt_outputs)
    print(trt_outputs.reshape((3, 960, 960)))
    print(trt_outputs)
    # print("cost time: ", time.time() - t1)


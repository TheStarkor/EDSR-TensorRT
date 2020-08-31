import tensorrt as trt

onnx_file_name = "edsr.onnx"
tensorrt_file_name = "edsr.plan"
fp_16_mode = True
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int) (trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

builder = trt.Builder(TRT_LOGGER)
print(1)
network = builder.create_network(EXPLICIT_BATCH)
print(2)
parser = trt.OnnxParser(network, TRT_LOGGER)
print(3)

builder.max_workspace_size = (1 << 10)
builder.fp16_mode = True

with open(onnx_file_name, 'rb') as model:
    print(4)
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

print(5)
engine = builder.build_cuda_engine(network)
print(6)
buf = engine.serialize()
print(7)
with open(tensorrt_file_name, 'wb') as f:
    f.write(buf)
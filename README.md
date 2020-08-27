# EDSR-TensorRT

### requirement
- nvidia-docker


## Start
```
$ docker run -d --gpus all -it --name <name> -v <home dir>:<container dir> nvcr.io/nvidia/tensorrt:20.08-py3
$ python -m pip install requirements.txt
```

1. Freezing a tensorflow model (in Colab)
2. `convert-to-uff edsr_000x00x00.pb` (in docker)
3. `python main.py`
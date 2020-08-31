# EDSR-TensorRT

### requirement
- nvidia-docker


## Start
```
$ docker run -d --gpus all -it --name <name> -v <home dir>:<container dir> nvcr.io/nvidia/tensorrt:20.08-py3
$ python -m pip install requirements.txt
```
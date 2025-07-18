# LLM Compression : Enhancing AWQ

Improved AWQ(Activation-aware Weight Quantization) with extra scaling (2024 Fall graduation project)

This work is based on MIT HAN Lab's [AWQ](https://github.com/mit-han-lab/llm-awq). 

## Updates

2024/12/19 : The code is uploaded.
2025/07/18 : Dockerfile is uploaded.

## Install

1. Clone this repository

```
git clone https://github.com/jundaree/ExtraAWQ.git
cd ExtraAWQ
```

2. (Optional) Open Dockerfile, and you can select CUDA 11.8 or CUDA 12.1(default) by commenting out
<br>
3. Create Docker image and container using the Dockerfile

```
docker build -t extraawq_i .
docker run --name extraawq \
   --env="DISPLAY" \
   --env="QT_X11_NO_MITSHM=1" \
   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   --gpus all --shm-size=64G -it extraawq_i
xhost +local:
```

## Download model checkpoints

```
cd scripts
sh download_checkpoints.sh
```

## Run ExtraAWQ 

1. (Optional) You can comment out the second loop in *opt_example.sh*,*llama_example.sh*,*llama2_example.sh*, as the pseudo-quantized model after the first loop is enough to show our results.

2. Run the shell file

```
sh opt_example.sh
sh llama_example.sh
sh llama2_example.sh
```

## Plot scales before ExtraAWQ and after ExtraAWQ


```
cd ../utils

python lineplot.py --llm opt
python lineplot.py --llm llama
python lineplot.py --llm Llama-2

python scatterplot.py --llm opt
python scatterplot.py --llm llama
python scatterplot.py --llm Llama-2
```


## Poster
![alt text](https://github.com/jundaree/ExtraAWQ/blob/main/poster.jpg?raw=true)


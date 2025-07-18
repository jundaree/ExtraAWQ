# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables to avoid user prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Update and install common packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3-pip

# enable PEP 660 support
RUN pip install --upgrade pip

RUN ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /workspace


RUN cd /workspace && \
    git clone https://github.com/jundaree/ExtraAWQ.git

# CUDA 12.1
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
# RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118


RUN cd /workspace/ExtraAWQ && \
    pip install -e .

RUN cd /workspace/ExtraAWQ/awq/kernels && \
    python setup.py install


# Default command
CMD ["/bin/bash"]

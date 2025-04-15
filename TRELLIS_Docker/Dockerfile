# Use an appropriate base image with Conda pre-installed
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS build

# Set environment variables for Conda and CUDA
ENV PATH="/opt/conda/bin:/usr/local/cuda-11.8/bin:$PATH" \
    CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    CUDA_HOME=/usr/local/cuda-11.8 \
    LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget build-essential python3-dev python3-pip git-lfs && \
    git lfs install && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda with Python 3.10
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Clone the TRELLIS repository
RUN git clone --recurse-submodules https://github.com/MontagueM/TRELLIS.git /app

# Initialize Conda in the shell
RUN /opt/conda/bin/conda init bash

# Create and configure the Conda environment
RUN bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda create -n trellis python=3.10 -y && \
    conda activate trellis && \
    pip install --upgrade pip wheel setuptools && \
    conda install -y pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 -c pytorch -c nvidia && \
    pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh xatlas pyvista pymeshfix igraph transformers flash-attn spconv-cu118 plyfile && \
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# Install Kaolin
RUN bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate trellis && \
    pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html"

# Install additional extensions
RUN bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate trellis && \
    mkdir -p /tmp/extensions && \
    git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast && \
    git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast && \
    pip install /tmp/extensions/diffoctreerast && \
    git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ && \
    cp -r /app/extensions/vox2seq /tmp/extensions/vox2seq && \
    pip install /tmp/extensions/vox2seq"

# Clone the Hugging Face model repository
RUN git clone https://huggingface.co/JeffreyXiang/TRELLIS-image-large /models/TRELLIS-image-large

# Set environment variables for caching
ENV TRANSFORMERS_CACHE=/app_data/.cache/transformers \
    HF_HOME=/app_data/.cache/huggingface \
    NUMBA_CACHE_DIR=/app_data/.cache/numba_cache \
    WARP_CACHE_DIR=/app_data/.cache/warp \
    TORCH_HOME=/app_data/.cache/torch \
    U2NET_HOME=/app_data/.cache/u2net

# Create cache directories
RUN mkdir -p $TRANSFORMERS_CACHE $HF_HOME $TORCH_HOME $U2NET_HOME /app/output && \
    chown -R 1000:1000 /app_data/.cache

# Copy application files
COPY scripts/image_to_3d.py /app/image_to_3d.py
COPY entrypoint.sh /app/entrypoint.sh
COPY .cache /app_data/.cache
RUN wget 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx' -O "${U2NET_HOME}/u2net.onnx"

# Set the working directory
WORKDIR /app

# Set default shell
SHELL ["/bin/bash", "-c"]

# Set default command
CMD ["/bin/bash"]
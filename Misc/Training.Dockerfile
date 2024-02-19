FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

# Setup Environment Variables
ENV CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# Setup System Utilities
RUN apt-get update --yes --quiet \
    && apt-get upgrade --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
        apt-utils \
        autoconf \
        automake \
        bc \
        build-essential \
        ca-certificates \
        check \
        cmake \
        curl \
        dmidecode \
        emacs \
        g++\
        gcc \
        git \
        iproute2 \
        jq \
        kmod \
        libaio-dev \
        libcurl4-openssl-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        libibverbs-dev \
        libnuma-dev \
        libnuma1 \
        libomp-dev \
        libsm6 \
        libssl-dev \
        libsubunit-dev \
        libsubunit0 \
        libtool \
        libxext6 \
        libxrender-dev \
        make \
        moreutils \
        net-tools \
        ninja-build \
        openssh-client \
        openssh-server \
        openssl \
        pkg-config \
        python3-dev \
        software-properties-common \
        sudo \
        unzip \
        util-linux \
        vim \
        wget \
        zlib1g-dev \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH=/root/.cargo/bin:$PATH

# Setup base Python to bootstrap Mamba
RUN add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-lib2to3 \
        python3.11-gdbm \
        python3.11-tk \
        pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 999 \
    && update-alternatives --config python3 \
    && ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# Setup optimized Pytorch Mamba environment
RUN wget -O /tmp/Miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/Miniforge.sh -b -p /Miniforge \
    && source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba update -y -q -n base -c defaults mamba \
    && mamba create -y -q -n pre-train python=3.11 \
    && mamba activate pre-train \
    && mamba install -y -q -c conda-forge \
        charset-normalizer \
        gputil \
        ipython \
        mkl \
        mkl-include \
        numpy \
        pandas \
        scikit-learn \
        wandb \
    && mamba install -y -q -c pytorch magma-cuda121 \
    && mamba install -y -q -c pytorch -c nvidia pytorch==2.1.2 pytorch-cuda=12.1 \
    && mamba clean -a -f -y

# Install Flash Attention
RUN source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba activate pre-train \
    && export MAX_JOBS=$(($(nproc) - 2)) \
    && pip install --no-cache-dir ninja packaging \
    && pip install flash-attn==2.4.2 --no-build-isolation

# Install optimized NVIDIA Apex
RUN source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba activate pre-train \
    && export MAX_JOBS=$(($(nproc) - 2)) \
    && tmp_apex_path="/tmp/apex" \
    && rm -rf $tmp_apex_path \
    && git clone https://github.com/NVIDIA/apex $tmp_apex_path \
    && cd $tmp_apex_path \
    && git checkout 23.08 \
    && pip install -v \
        --disable-pip-version-check \
        --no-cache-dir \
        --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" \
        --config-settings "--build-option=--permutation_search" \
        --config-settings "--build-option=--bnp" \
        --config-settings "--build-option=--xentropy" \
        --config-settings "--build-option=--focal_loss" \
        --config-settings "--build-option=--index_mul_2d" \
        --config-settings "--build-option=--deprecated_fused_adam" \
        --config-settings "--build-option=--deprecated_fused_lamb" \
        --config-settings "--build-option=--fast_layer_norm" \
        --config-settings "--build-option=--fmha" \
        --config-settings "--build-option=--fast_multihead_attn" \
        --config-settings "--build-option=--transducer" \
        --config-settings "--build-option=--nccl_p2p" ./

# Install pre-compiled Deepspeed
RUN source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba activate pre-train \
    && export MAX_JOBS=$(($(nproc) - 2)) \
    && pip install deepspeed-kernels \
    && DS_BUILD_AIO=1 \
        DS_BUILD_CCL_COMM=1 \
        DS_BUILD_CPU_ADAGRAD=1 \
        DS_BUILD_FUSED_LAMB=1 \
        DS_BUILD_CPU_ADAM=1 \
        DS_BUILD_FUSED_ADAM=1 \
        DS_BUILD_UTILS=1 \
        python3 -m pip install deepspeed==0.12.6 -v \
            --disable-pip-version-check \
            --no-cache \
            --global-option="build_ext" \
            --global-option="-j"$(($(nproc) - 2))

# Install HuggingFace and LM/Quantization dependencies
RUN source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba activate pre-train \
    && pip install transformers==4.36.2 \
        tokenizers \
        datasets \
        evaluate \
        accelerate \
        peft \
        protobuf \
        sentencepiece!=0.1.92 \
        rouge-score!=0.0.7,!=0.0.8,!=0.1,!=0.1.1 \
        nltk \
        py7zr \
        seqeval \
        requests \
        tqdm \
        bitsandbytes==0.42.0 \
        autoawq==0.1.8
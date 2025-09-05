FROM nvcr.io/nvidia/pytorch:24.04-py3
# Install build dependencies
RUN apt-get update && apt-get install -y \
    git build-essential \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and build QVina 2
# Clone and build QVina 2
RUN cd /tmp && \
    git clone https://github.com/QVina/qvina.git && \
    cd qvina && \
    ls -la && \
    # The actual makefile might be in src/ directory
    if [ -f src/Makefile ]; then \
        cd src && \
        make && \
        cp vina /usr/local/bin/qvina02; \
    elif [ -f Makefile ]; then \
        make && \
        cp qvina2 /usr/local/bin/qvina02 || cp vina /usr/local/bin/qvina02; \
    else \
        echo "No Makefile found, checking structure..." && \
        find . -name "Makefile" -o -name "*.cpp"; \
    fi && \
    cd / && rm -rf /tmp/qvina
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install \
    datasets transformers dacite pyyaml numpy packaging safetensors \
    selfies tqdm dataclasses rdkit tokenizers atomInSmiles \
    "jsonargparse[signatures]>=4.27.7" lightning torchdata einops openbabel-wheel wandb

# Don't set WORKDIR or copy files to /app
# Just install the packages
RUN pip install --no-dependencies -U pytdc==1.1.15 fuzzywuzzy huggingface-hub
COPY setup.py /tmp/setup.py
COPY README.md /tmp/README.md
COPY requirements.txt /tmp/requirements.txt
COPY in_virtuo_gen /tmp/in_virtuo_gen
COPY in_virtuo_reinforce /tmp/in_virtuo_reinforce

WORKDIR /tmp
RUN pip install -e .
WORKDIR /
# Create a generic qvina02 wrapper in /usr/local/bin
RUN printf '#!/bin/bash\nexec /usr/bin/vina "$@"\n' > /usr/local/bin/qvina02 && \
    chmod +x /usr/local/bin/qvina02
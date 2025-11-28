FROM nvcr.io/nvidia/pytorch:25.02-py3

# setup
RUN apt-get update && apt-get install python3-pip python3-venv -y
RUN pip install --upgrade pip setuptools==70.3.0
    
# Install the rest of dependencies.
RUN pip install \
    torchao==0.12.0 \
    torchtune==0.6.1 \
    pyarrow==18.1.0 \
    datasets==3.6.0 \
    openai==2.8.1 \
    tqdm>=4.65 \
    transformers \
    accelerate \
    wandb \
    dacite \
    pyyaml \
    numpy \ 
    packaging \
    safetensors \
    sentencepiece \
    tensorboard \
    pandas \
    jupyter \
    deepspeed \
    seaborn

# Create a work directory
RUN mkdir -p /workspace
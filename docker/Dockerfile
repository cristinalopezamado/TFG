# Imaxe base de CUDA 11.1 con Ubuntu 20.04
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Ferramentas básicas e dependencias
RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libgl1 \
    libglib2.0-0
# pip para Python 3.8
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py
    
#torch e torchvision
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --upgrade torch torchvision

# Paquetes adicionais necesarios para Detectron2 e omegaconf
RUN pip install cython pyyaml==5.4.1 matplotlib opencv-python omegaconf scikit-learn

# Instalamos librerías adicionais necesarias para algunhas operacións de Detectron2
RUN apt-get install -y libopencv-dev

# Instalamos Detectron2
RUN python3.8 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Clonamos o repositorio TFG dende GitHub
RUN git clone https://github.com/cristinalopezamado/TFG.git /TFG

#PYTHONPATH ten a ubicación do código
ENV PYTHONPATH /detectron2:/TFG/unbiased-teacher-v2:$PYTHONPATH
CMD ["python3"]

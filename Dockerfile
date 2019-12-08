FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime as base

# Input data and output artifacts
VOLUME [ "/data", "/artifacts" ]

# Expose Tensorboard ports
EXPOSE 6006/tcp 6006/udp

COPY [ "docker/run.sh", "docker/entrypoint.sh", "/" ]
ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/run.sh" ]

# Required for ROUGE
RUN apt-get -y update && apt-get install -y git libxml-parser-perl

WORKDIR /app

#RUN git clone https://github.com/NVIDIA/apex
#RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex/

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY bert_config_uncased_base.json /app/

COPY ./src /app

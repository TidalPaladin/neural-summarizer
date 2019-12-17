ARG upstream=1.3-cuda10.1-cudnn7-runtime
FROM pytorch/pytorch:${upstream} as base

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt

# Required for ROUGE
RUN apt-get -y update && apt-get install -y libxml-parser-perl

COPY [ "docker/run.sh", "docker/entrypoint.sh", "/" ]
COPY bert_config_uncased_base.json /app/

# Input data and output artifacts
VOLUME [ "/data", "/artifacts" ]

# Expose Tensorboard ports
EXPOSE 6006/tcp 6006/udp

ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/run.sh" ]

COPY ./src /app

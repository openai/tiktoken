FROM python:3.7.10-slim

# Setup user
ARG CONTAINER_USER_ID=1000
RUN getent group ${CONTAINER_USER_ID} || groupadd -g ${CONTAINER_USER_ID} tiktoken
RUN useradd tiktoken -u ${CONTAINER_USER_ID} -g ${CONTAINER_USER_ID} -o -m -s /shell/bash

WORKDIR /mnt/

# Install packages
# libmagic1 required for running tests
RUN apt-get update && apt-get install -y libmagic1 curl git build-essential

# Setup Python
ENV PYTHONUNBUFFERED=1
ENV PATH="${PATH}:/home/tiktoken/.local/bin"

RUN pip3 install --upgrade pip

USER tiktoken

# Install Rust (needed for tiktokenizer py dep)
RUN curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | bash -s - -y

CMD [ "/bin/bash", "-c", "sleep infinity" ]

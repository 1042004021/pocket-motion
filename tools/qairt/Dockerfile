FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y -qq \
      python3 python3-venv python3-pip \
      libc++1 libc++abi1 libunwind8 \
      clang ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/qvenv && \
    /opt/qvenv/bin/pip install --no-cache-dir \
      onnx==1.14.1 \
      numpy==1.26.4 \
      pyyaml \
      packaging \
      pandas \
      onnxsim \
      onnxruntime==1.17.3

ENV PYTHONPATH=/qairt/qairt/2.39.0.250925/lib/python
ENV PATH=/opt/qvenv/bin:$PATH

WORKDIR /work
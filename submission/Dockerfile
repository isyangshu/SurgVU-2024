FROM python:3.11

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean -y

RUN apt-get update && apt-get upgrade -y
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 libx264-dev

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
RUN python -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm model.py /opt/algorithm/
COPY --chown=algorithm:algorithm checkpoint-best.pth /opt/algorithm/
COPY --chown=algorithm:algorithm video_transforms.py /opt/algorithm/
COPY --chown=algorithm:algorithm volume_transforms.py /opt/algorithm/
COPY --chown=algorithm:algorithm functional.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@

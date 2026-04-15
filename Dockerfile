FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN usermod -l carla ubuntu
RUN groupmod -n carla ubuntu

COPY --from=docker.io/tonychi/carla:0.9.16 --chown=carla:carla /opt/carla /opt/carla

RUN <<EOF
    apt update
    apt install -y \
        git ca-certificates \
        build-essential g++-12 cmake ninja-build libvulkan1 \
        python3 python3-dev python3-pip python3-venv autoconf \
        wget curl rsync unzip git git-lfs libpng-dev libtiff5-dev \
        libjpeg-dev
    rm -rf /var/lib/apt/lists/*
EOF

ADD https://security.ubuntu.com/ubuntu/pool/main/t/tiff/libtiff5_4.3.0-6_amd64.deb /tmp/libtiff5-dev.deb
RUN dpkg -i /tmp/libtiff5-dev.deb && rm /tmp/libtiff5-dev.deb

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ADD https://github.com/carla-simulator/scenario_runner.git /opt/scenario_runner

USER carla
WORKDIR /app
COPY --chown=carla:carla ./pyproject.toml .
COPY --chown=carla:carla ./uv.lock .
RUN uv sync --locked
RUN uv add /opt/carla/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-linux_x86_64.whl
RUN uv add -r /opt/scenario_runner/requirements.txt
ENV PYTHONPATH=/opt/scenario_runner/:/opt/carla/PythonAPI/carla/
COPY . .

ENV PORT=50051
ENV CARLA_PORT=2000
ENV CARLA_TIMEOUT=5

ENTRYPOINT [ "/bin/bash" ]
CMD [ "/app/entrypoint.sh" ]

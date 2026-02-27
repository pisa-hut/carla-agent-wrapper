just:
    docker build . -t tonychi/carla-agent-wrapper
    docker run -it --rm --gpus all \
    --network host \
    --runtime=nvidia \
    --env=NVIDIA_VISIBLE_DEVICES=all \
    --env=NVIDIA_DRIVER_CAPABILITIES=all \
    -v /opt/sbsvf/map/tyms/xodr:/mnt/map/xodr \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority \
    -e DISPLAY -e PORT=50051 -e CARLA_HOST=localhost -e CARLA_PORT=2000 \
    tonychi/carla-agent-wrapper:latest

enter:
    docker build . -t tonychi/carla-agent-wrapper
    docker run -it --rm --gpus all \
    --network host \
    -v /opt/sbsvf/map/tyms/xodr:/mnt/map/xodr \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority \
    -e DISPLAY -e PORT=50051 -e CARLA_HOST=localhost -e CARLA_PORT=2000 \
    --entrypoint bash \
    tonychi/carla-agent-wrapper:latest

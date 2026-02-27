#!/bin/bash

pushd /opt/carla
CMD="./CarlaUE4.sh -carla-port=${CARLA_PORT:-2000} -RenderOffScreen -nosound -quality-level=Low" 
echo "Running command: $CMD"
eval $CMD &
popd
 
pushd /app
uv run carla_agent_wrapper/server.py
popd

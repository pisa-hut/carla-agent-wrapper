#!/bin/bash
pushd /app
uv run carla_agent_wrapper/server.py
popd

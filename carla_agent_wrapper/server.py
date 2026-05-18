from pisa_api.av import serve_av_system
from pisa_api.wrapper import setup_logging

try:
    from .carla_agent import CarlaAgentAV
except ImportError:
    from carla_agent import CarlaAgentAV

setup_logging()


if __name__ == "__main__":
    serve_av_system(CarlaAgentAV(), name="CARLA-Agent")

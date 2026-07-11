from pisa_api.av import serve_av_system
from pisa_api.wrapper import setup_logging

try:
    from .carla_agent import CarlaAgentAV
    from .version import wrapper_version
except ImportError:
    from carla_agent import CarlaAgentAV
    from version import wrapper_version

setup_logging()


if __name__ == "__main__":
    serve_av_system(
        CarlaAgentAV(),
        name="carla-agent-wrapper",
        version=wrapper_version(),
    )

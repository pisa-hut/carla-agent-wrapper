import logging

import grpc
from carla_agent import CarlaAgentAV
from google.protobuf.json_format import MessageToDict
from pisa_api import av_server_pb2
from pisa_api.empty_pb2 import Empty
from pisa_api.wrapper import BaseAvServer, serve_av, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class AVServer(BaseAvServer):
    _name = "CARLA-Agent"

    def __init__(self):
        super().__init__()
        self._av = CarlaAgentAV()

    def Init(self, request, context):
        config = MessageToDict(request.config.config)
        output_dir = request.output_dir.path
        map_name = request.map_name
        logger.debug("Init config: %s", config)

        try:
            self._av.init(config, output_dir, map_name)
            return av_server_pb2.AvServerMessages.InitResponse(
                success=True, msg="Initialization successful"
            )
        except Exception as e:
            logger.exception("Failed to initialize AV in Init")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Carla-Agent Initialization failed: {str(e)}")
            return av_server_pb2.AvServerMessages.InitResponse(
                success=False, msg=f"Initialization failed: {str(e)}"
            )

    def Reset(self, request, context):
        output_dir = request.output_dir.path
        scenario_pack = request.scenario_pack
        initial_observation = request.initial_observation
        return av_server_pb2.AvServerMessages.ResetResponse(
            ctrl_cmd=self._av.reset(output_dir, scenario_pack, initial_observation)
        )

    def Step(self, request, context):
        observation = request.observation
        timestamp_ns = request.timestamp_ns
        return av_server_pb2.AvServerMessages.StepResponse(
            ctrl_cmd=self._av.step(observation, timestamp_ns)
        )

    def Stop(self, request, context):
        if self._av is not None:
            self._av.stop()
        return Empty()

    def ShouldQuit(self, request, context):
        should_quit = self._av.should_quit()
        return av_server_pb2.AvServerMessages.ShouldQuitResponse(should_quit=should_quit)


if __name__ == "__main__":
    serve_av(AVServer(), name="CARLA-Agent")

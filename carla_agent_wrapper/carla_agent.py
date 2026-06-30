import contextlib
import logging
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent
from pisa_api.av import (
    AvError,
    AvPreconditionFailed,
    AvTimeout,
    AvUnavailable,
    ControlCommand,
    ControlMode,
    InitRequest,
    InvalidAvRequest,
    ObjectStateData,
    ObservationData,
    ResetRequest,
    ResetResponse,
    RoadObjectType,
    ScenarioPackData,
    ShapeType,
    ShouldQuitResponse,
    StepRequest,
    StepResponse,
)

try:
    from .lifecycle import clear_dynamic_actors, destroy_actor, force_async_world_for_cleanup
except ImportError:
    from lifecycle import clear_dynamic_actors, destroy_actor, force_async_world_for_cleanup

logger = logging.getLogger(__name__)

VALID_BEHAVIORS = {"cautious", "normal", "aggressive"}
VALID_AGENT_TYPES = {"behavior", "basic", "constant_velocity", "constant-velocity"}

BLUEPRINT_CANDIDATES = {
    RoadObjectType.PEDESTRIAN: ("walker.pedestrian.0001", "walker.pedestrian.*", "walker.*"),
    RoadObjectType.BUS: ("vehicle.mitsubishi.fusorosa", "vehicle.*bus*", "vehicle.*"),
    RoadObjectType.TRUCK: ("vehicle.carlamotors.carlacola", "vehicle.*truck*", "vehicle.*"),
    RoadObjectType.SEMITRAILER: ("vehicle.carlamotors.carlacola", "vehicle.*truck*", "vehicle.*"),
    RoadObjectType.TRAILER: ("vehicle.carlamotors.firetruck", "vehicle.*truck*", "vehicle.*"),
    RoadObjectType.VAN: ("vehicle.mercedes.sprinter", "vehicle.*van*", "vehicle.*"),
    RoadObjectType.MOTORCYCLE: ("vehicle.vespa.zx125", "vehicle.*", "vehicle.*"),
    RoadObjectType.BICYCLE: ("vehicle.bh.crossbike", "vehicle.*bike*", "vehicle.*"),
    RoadObjectType.TRAIN: ("vehicle.*",),
    RoadObjectType.TRAM: ("vehicle.*",),
    RoadObjectType.WHEEL_CHAIR: ("walker.pedestrian.*", "walker.*"),
    RoadObjectType.ANIMAL: ("walker.pedestrian.*", "walker.*"),
    RoadObjectType.CAR: ("vehicle.*",),
    RoadObjectType.UNKNOWN: ("vehicle.*",),
}


class CarlaAgentAV:
    """
    CARLA automatic-control style AV adapter.
    - init(): connect to CARLA, prepare agent classes
    - reset(): find ego by role_name, create agent, set destination/speed
    - step(): run agent.run_step(), convert to ControlCommand
    """

    def __init__(self):
        self._carla = carla
        self._BehaviorAgent = BehaviorAgent
        self._BasicAgent = BasicAgent
        self._ConstantVelocityAgent = ConstantVelocityAgent

        self._spawned_actor_ids = set()
        self._loaded_map_name = None
        self._loaded_opendrive_path = None

        self._client = None
        self._server_version = None
        self._server_process = None
        self._finalized = True

        self._quit_flag = False
        self._quit_msg = ""

        self._server_log_path = "/mnt/output/carla_server"
        os.makedirs(self._server_log_path, exist_ok=True)
        # subprocess.Popen dups these descriptors into the child, so it's
        # safe to close the parent handles after Popen returns — the
        # CARLA process keeps writing through its own fds.
        with (
            open(f"{self._server_log_path}/stdout.log", "w") as out,
            open(f"{self._server_log_path}/stderr.log", "w") as err,
        ):
            self._server_process = subprocess.Popen(
                ["/app/carla_server.sh"],
                stdout=out,
                stderr=err,
            )
        logger.info("CARLA service launched.")

        # reset
        self._world = None
        self._map = None
        self._vehicle = None
        self._agent = None
        self._other_actors: list[Any] = []
        self._other_actor_types: list[RoadObjectType] = []
        self._other_actors_by_key: dict[Any, Any] = {}
        self._other_actor_types_by_key: dict[Any, RoadObjectType] = {}
        self._using_tracking_ids = False

    def _connect(self, timeout: float = 2.0):
        if self._server_version is not None:
            return
        logger.debug("Connecting to CARLA...")
        self._client = carla.Client(
            os.environ.get("CARLA_HOST", "localhost"),
            int(os.environ.get("CARLA_PORT", 2000)),
        )
        try:
            self._client.set_timeout(timeout)
            self._server_version = self._client.get_server_version()
        finally:
            self._client.set_timeout(float(os.environ.get("CARLA_TIMEOUT", 10.0)))
        logger.debug("Connected to CARLA")

    def _ensure_connected(self) -> bool:
        config = getattr(self, "config", {}) or {}
        if "carla_connect_timeout_seconds" in config:
            timeout = self._config_float("carla_connect_timeout_seconds", 30.0)
        else:
            timeout = self._config_float("max_retry_times", 15.0) * 2
        retry_interval = self._config_float("retry_interval_seconds", 2.0)
        if timeout <= 0:
            raise InvalidAvRequest("carla_connect_timeout_seconds must be positive")
        if retry_interval <= 0:
            raise InvalidAvRequest("retry_interval_seconds must be positive")
        end_time = time.time() + timeout

        while self._server_version is None:
            try:
                self._connect(2.0)
                return True
            except Exception:
                remaining = end_time - time.time()
                if remaining <= 0:
                    logger.exception("Failed to connect to CARLA: connection timeout.")
                    return False
                logger.exception(
                    "Failed to connect to CARLA, retrying in %.1f seconds...",
                    retry_interval,
                )
                time.sleep(retry_interval)

        return True

    def _extract_xyz(self, pos) -> tuple[float, float, float]:
        if pos is None:
            raise ValueError("Position is None")
        world = getattr(pos, "world", None)
        source = world if world is not None else pos
        missing = [name for name in ("x", "y", "z") if not hasattr(source, name)]
        if missing:
            raise ValueError(
                f"Position object is missing coordinate field(s): {', '.join(missing)}. "
                f"position={pos!r}"
            )
        try:
            return float(source.x), float(source.y), float(source.z)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Failed to convert position coordinates to float: {pos!r}") from exc

    def _extract_yaw(self, pos) -> float:
        if pos is None:
            return 0.0
        if hasattr(pos, "yaw"):
            try:
                return float(pos.yaw)
            except (TypeError, ValueError) as exc:
                raise InvalidAvRequest(f"Position yaw must be a float, got {pos.yaw!r}") from exc
        world = getattr(pos, "world", None)
        if world is not None and hasattr(world, "h"):
            try:
                return float(world.h)
            except (TypeError, ValueError) as exc:
                raise InvalidAvRequest(
                    f"Position heading must be a float, got {world.h!r}"
                ) from exc
        if hasattr(pos, "h"):
            try:
                return float(pos.h)
            except (TypeError, ValueError) as exc:
                raise InvalidAvRequest(f"Position heading must be a float, got {pos.h!r}") from exc
        return 0.0

    def _to_carla_location(self, pos) -> Any:
        try:
            x, y, z = self._extract_xyz(pos)
        except ValueError as exc:
            raise ValueError(f"Failed to convert position to CARLA location: {pos!r}") from exc
        y = float(y) * self._coordinate_y_sign
        return self._carla.Location(
            x=float(x),
            y=y,
            z=float(z),
        )

    def _get_target_speed_kmh(self, sps: ScenarioPackData) -> float:
        speed = self._target_speed
        if speed is None and sps is not None:
            ego = getattr(sps, "ego", None)
            if ego is not None and getattr(ego, "target_speed", None) is not None:
                speed = ego.target_speed
        if speed is None:
            speed = 0.0
        try:
            speed = float(speed)
        except (TypeError, ValueError) as exc:
            raise InvalidAvRequest(f"target_speed must be a float, got {speed!r}") from exc
        if self._target_speed_is_mps:
            speed = speed * 3.6
        return speed

    def _validate_config(self) -> None:
        if self._agent_type not in VALID_AGENT_TYPES:
            raise InvalidAvRequest(f"Unsupported CARLA agent_type: {self._agent_type!r}")
        if self._agent_type == "behavior" and self._behavior not in VALID_BEHAVIORS:
            raise InvalidAvRequest(
                f"Unsupported CARLA behavior: {self._behavior!r}. "
                f"Expected one of: {', '.join(sorted(VALID_BEHAVIORS))}"
            )

    def _config_sign(self, name: str, default: float) -> float:
        value = self._config_float(name, default)
        if abs(value) < 1e-6:
            raise InvalidAvRequest(f"{name} must be non-zero")
        return 1.0 if value > 0 else -1.0

    def _config_float(self, name: str, default: float) -> float:
        raw = self.config.get(name, default)
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise InvalidAvRequest(f"{name} must be a float, got {raw!r}") from exc

    def _build_agent(self, target_speed_kmh: float):
        if self._vehicle is None:
            return None

        agent_opt_dict = {
            "sampling_resolution": self._route_sampling_resolution,
            "base_min_distance": self._local_planner_base_min_distance,
            "distance_ratio": self._local_planner_distance_ratio,
        }
        if self._agent_type == "behavior":
            agent = self._BehaviorAgent(
                self._vehicle,
                behavior=self._behavior,
                opt_dict=agent_opt_dict,
                map_inst=self._map,
            )
            self._configure_local_planner(agent)
            logger.info(f"Initialized CARLA BehaviorAgent with behavior={self._behavior}")
            logger.info(f"Target speed: {target_speed_kmh} km/h")

        elif self._agent_type == "basic":
            agent = self._BasicAgent(
                self._vehicle,
                target_speed=target_speed_kmh,
                opt_dict=agent_opt_dict,
                map_inst=self._map,
            )
            self._configure_local_planner(agent)
            logger.info(f"Initialized CARLA BasicAgent with target_speed_kmh={target_speed_kmh}")
        elif self._agent_type in ("constant_velocity", "constant-velocity"):
            agent = self._ConstantVelocityAgent(
                self._vehicle,
                target_speed=target_speed_kmh,
                opt_dict=agent_opt_dict,
                map_inst=self._map,
            )
        else:
            raise InvalidAvRequest(f"Unsupported CARLA agent_type: {self._agent_type!r}")

        agent.set_target_speed(target_speed_kmh)
        if hasattr(agent, "follow_speed_limits"):
            agent.follow_speed_limits(self._follow_speed_limits)
        if hasattr(agent, "ignore_traffic_lights"):
            agent.ignore_traffic_lights(self._ignore_traffic_lights)
        if hasattr(agent, "ignore_stop_signs"):
            agent.ignore_stop_signs(self._ignore_stop_signs)
        if hasattr(agent, "ignore_vehicles"):
            agent.ignore_vehicles(self._ignore_vehicles)
        return agent

    def _configure_local_planner(self, agent) -> None:
        local_planner = getattr(agent, "_local_planner", None)
        if local_planner is None:
            return

        local_planner._base_min_distance = self._local_planner_base_min_distance
        local_planner._distance_ratio = self._local_planner_distance_ratio
        local_planner._min_distance = self._local_planner_base_min_distance

    def init(self, request: InitRequest) -> None:
        self._output_dir = request.output_dir
        self.config = request.config or {}
        self._fixed_delta_seconds = request.dt

        if not self._finalized:
            self._finalize()

        self._sync = bool(self.config.get("sync", True))
        self._no_rendering = bool(self.config.get("no_rendering", True))

        self._ego_role_name = self.config.get("ego_role_name", "hero")
        self._ego_bp_id = self.config.get("ego_bp_id", "vehicle.tesla.model3")
        self._agent_type = str(self.config.get("agent_type", "behavior")).lower()
        self._behavior = str(self.config.get("behavior", "normal")).lower()
        self._validate_config()
        self._random_destination = bool(self.config.get("random_destination", False))
        self._follow_speed_limits = bool(self.config.get("follow_speed_limits", False))
        self._ignore_traffic_lights = bool(self.config.get("ignore_traffic_lights", False))
        self._ignore_stop_signs = bool(self.config.get("ignore_stop_signs", False))
        self._ignore_vehicles = bool(self.config.get("ignore_vehicles", False))

        self._target_speed = self.config.get("target_speed", None)
        self._target_speed_is_mps = bool(self.config.get("target_speed_is_mps", False))
        self._local_planner_base_min_distance = self._config_float(
            "local_planner_base_min_distance", 3.0
        )
        self._local_planner_distance_ratio = self._config_float("local_planner_distance_ratio", 0.5)
        self._route_sampling_resolution = self._config_float("route_sampling_resolution", 3.0)

        legacy_yaw_sign = self._config_sign("yaw_sign", -1.0)
        self._coordinate_y_sign = self._config_sign("coordinate_y_sign", legacy_yaw_sign)
        self._yaw_sign = self._config_sign("yaw_sign", legacy_yaw_sign)
        self._steer_sign = self._config_sign("steer_sign", legacy_yaw_sign)
        self._yaw_offset_deg = self._config_float("yaw_offset_deg", 0.0)
        self._spawn_z_offset = self._config_float("spawn_z_offset", 3.0)
        self._xodr_root = Path(self.config.get("xodr_root", "/mnt/map/xodr"))
        self._reuse_generated_world = bool(self.config.get("reuse_generated_world", True))
        self._traffic_manager_port = int(os.environ.get("CARLA_TM_PORT", 8000))
        self._manage_traffic_manager_sync = bool(
            self.config.get("manage_traffic_manager_sync", False)
        )
        if not self._ensure_connected():
            raise AvTimeout("Timed out connecting to CARLA")

        self._prepare_reused_server_state()
        self._quit_flag = False
        self._quit_msg = ""
        return None

    def reset(
        self,
        request: ResetRequest,
    ) -> ResetResponse:
        if not self._finalized:
            self._finalize()

        self._finalized = False
        self._output_dir = request.output_dir
        # os.makedirs(self._output_dir, exist_ok=True)
        sps = request.scenario_pack
        init_obs = request.initial_observation
        self._sps = sps
        self._quit_flag = False
        self._quit_msg = ""

        try:
            if sps is None:
                raise InvalidAvRequest("ScenarioPack is required to prepare CARLA agent")
            self._ensure_world(sps.map_name)
            self._clear_dynamic_actors()
            self._reset_other_actor_state()
            self._vehicle = None
            logger.debug("Ego vehicle found: %s", self._vehicle)
            if self._vehicle is None:
                self._vehicle = self._spawn_ego(init_obs, sps)

            self._apply_world_settings()

            target_speed_kmh = self._get_target_speed_kmh(sps)
            self._agent = self._build_agent(target_speed_kmh)
            if self._agent is None:
                raise RuntimeError("Failed to create CARLA agent")

            if self._random_destination:
                if self._map is None:
                    raise InvalidAvRequest("CARLA map not available for destination picking")
                spawn_points = self._map.get_spawn_points()
                if not spawn_points:
                    raise InvalidAvRequest("No spawn points available for random destination")
                dest = random.choice(spawn_points).location
            else:
                goal_pos = sps.ego.goal_config.position if sps is not None else None
                if goal_pos is None:
                    if self._map is None:
                        raise InvalidAvRequest("Goal position missing and CARLA map not available")
                    logger.warning(
                        "Goal position missing; using random destination from spawn points."
                    )
                    spawn_points = self._map.get_spawn_points()
                    if not spawn_points:
                        raise InvalidAvRequest("No spawn points available for destination fallback")
                    dest = random.choice(spawn_points).location
                else:
                    try:
                        dest = self._to_carla_location(goal_pos)
                    except ValueError as exc:
                        raise InvalidAvRequest(str(exc)) from exc
            dest.z += self._spawn_z_offset  # to avoid underground issues

            start_transform = self._vehicle.get_transform()
            start_wp = self._snap_to_waypoint(start_transform.location, "ego start")
            end_wp = self._snap_to_waypoint(dest, "destination")
            logger.debug(
                "Route start: %s, yaw: %.3f, snapped to: %s",
                start_transform.location,
                self._from_carla_yaw(start_transform.rotation.yaw),
                start_wp.transform.location,
            )
            logger.debug(
                "Route destination: %s, snapped to: %s",
                dest,
                end_wp.transform.location,
            )
            logger.info(f"Route start: {start_wp.transform.location}")
            logger.info(f"Route destination: {end_wp.transform.location}")
            try:
                self._agent.set_destination(end_wp.transform.location, start_wp.transform.location)
                self._ensure_route_ends_at_waypoint(end_wp)
            except Exception as exc:
                raise AvPreconditionFailed(
                    f"CARLA agent_type={self._agent_type!r} failed to plan a route. "
                    f"start={self._format_waypoint(start_wp)}, "
                    f"destination={self._format_waypoint(end_wp)}"
                ) from exc

            return ResetResponse(ctrl_cmd=self.step(StepRequest(observation=init_obs)).ctrl_cmd)
        except AvError:
            logger.exception("Failed to reset CARLA agent; finalizing partial state")
            self._finalize()
            raise

    def step(self, request: StepRequest) -> StepResponse:
        obs = request.observation
        if obs is None or getattr(obs, "ego", None) is None:
            raise InvalidAvRequest("Step observation must include ego state")
        self._update_and_tick(obs)
        if self._agent is None:
            raise RuntimeError("CARLA agent is not initialized")

        control = self._agent.run_step()
        if hasattr(self._agent, "done") and self._agent.done():
            self._quit_flag = True
            self._quit_msg = "CARLA agent reached the destination."

        steer_sv = float(control.steer) / getattr(self, "_steer_sign", 1.0)
        return StepResponse(
            ctrl_cmd=ControlCommand(
                mode=ControlMode.THROTTLE_STEER_BREAK,
                payload={
                    "throttle": float(control.throttle),
                    "brake": float(control.brake),
                    "steer": steer_sv,
                },
            )
        )

    def stop(self) -> None:
        self._finalize()
        self._quit_msg = "CARLA service stopped."
        self._terminate_server_process()
        self._client = None
        self._server_version = None
        self._world = None
        self._map = None
        self._loaded_map_name = None
        self._loaded_opendrive_path = None
        logger.info("CARLA service stopped.")

    def _finalize(self) -> None:
        try:
            self._destroy_spawned_actors()
        except Exception:
            logger.exception("Failed to destroy spawned actors")
        self._agent = None
        self._vehicle = None
        self._quit_flag = True
        self._quit_msg = "CARLA agent finalized."
        self._finalized = True

    def _terminate_server_process(self) -> None:
        process = self._server_process
        if process is None:
            return
        if process.poll() is not None:
            self._server_process = None
            return
        try:
            process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("CARLA server did not terminate in time; killing it")
            process.kill()
            process.wait(timeout=10)
        except Exception:
            logger.exception("Failed to terminate CARLA server process")
        finally:
            self._server_process = None

    def should_quit(self) -> ShouldQuitResponse:
        return ShouldQuitResponse(
            should_quit=self._quit_flag,
            msg=getattr(self, "_quit_msg", ""),
        )

    def _ensure_route_ends_at_waypoint(self, end_wp) -> None:
        local_planner = getattr(self._agent, "_local_planner", None)
        route_queue = getattr(local_planner, "_waypoints_queue", None)
        if route_queue is None:
            return

        end_loc = end_wp.transform.location
        road_option = None
        if route_queue:
            last_wp, road_option = route_queue[-1]
            if self._location_distance(last_wp.transform.location, end_loc) < 1e-3:
                return
        else:
            road_option = getattr(local_planner, "target_road_option", None)

        route_queue.append((end_wp, road_option))

    @staticmethod
    def _location_distance(a, b) -> float:
        if hasattr(a, "distance"):
            return float(a.distance(b))
        return math.sqrt(
            (float(a.x) - float(b.x)) ** 2
            + (float(a.y) - float(b.y)) ** 2
            + (float(a.z) - float(b.z)) ** 2
        )

    def _spawn_ego(self, init_obs: ObservationData, sps: ScenarioPackData):
        if self._world is None:
            raise AvUnavailable("CARLA world not available")

        bp_lib = self._world.get_blueprint_library()
        ego_bp = self._find_blueprint(bp_lib, (self._ego_bp_id, "vehicle.*"))
        if ego_bp is None:
            raise InvalidAvRequest("No vehicle blueprints available in CARLA")

        if ego_bp.has_attribute("role_name"):
            ego_bp.set_attribute("role_name", self._ego_role_name)

        pos = self._get_spawn_position(init_obs, sps)
        if pos is None:
            raise InvalidAvRequest("No spawn position available for ego vehicle")
        try:
            carla_pos = self._to_carla_location(pos)
        except ValueError as exc:
            raise InvalidAvRequest(str(exc)) from exc
        carla_pos.z += (
            self._spawn_z_offset
        )  # to avoid spawning underground due to map height issues
        carla_rot = self._carla.Rotation(
            pitch=0.0,
            yaw=self._to_carla_yaw(self._extract_yaw(pos)),
            roll=0.0,
        )
        transform = self._carla.Transform(carla_pos, carla_rot)
        ego = self._world.try_spawn_actor(ego_bp, transform)
        if ego is None:
            logger.warning("Initial spawn failed, trying spawn points...")
            spawn_points = self._world.get_map().get_spawn_points()
            if not spawn_points:
                raise AvPreconditionFailed("Failed to spawn ego vehicle (no spawn points)")
            ego = self._world.try_spawn_actor(ego_bp, spawn_points[0])
            if ego is None:
                raise AvPreconditionFailed("Failed to spawn ego vehicle")
        self._spawned_actor_ids.add(ego.id)

        try:
            phys = ego.get_physics_control()
            max_steer = max([w.max_steer_angle for w in phys.wheels])
            self._max_steer_rad = math.radians(max_steer)
        except Exception:
            self._max_steer_rad = None

        logger.debug("Ego vehicle spawned at %s with yaw %.3f", carla_pos, self._extract_yaw(pos))
        return ego

    def _to_carla_yaw(self, yaw_rad: float) -> float:
        return self._yaw_sign * math.degrees(yaw_rad) + self._yaw_offset_deg

    def _from_carla_yaw(self, yaw_deg: float) -> float:
        return math.radians((yaw_deg - self._yaw_offset_deg) * self._yaw_sign)

    def _snap_to_waypoint(self, location, label: str):
        if self._map is None:
            raise InvalidAvRequest("CARLA map not available for route planning")
        waypoint = self._map.get_waypoint(location, project_to_road=True)
        if waypoint is None:
            raise AvPreconditionFailed(
                f"Failed to project {label} location onto CARLA map: {location}"
            )
        return waypoint

    def _format_waypoint(self, waypoint) -> str:
        loc = waypoint.transform.location
        return (
            f"road_id={waypoint.road_id}, section_id={waypoint.section_id}, "
            f"lane_id={waypoint.lane_id}, s={waypoint.s:.3f}, "
            f"location=({loc.x:.3f}, {loc.y:.3f}, {loc.z:.3f})"
        )

    def _ensure_world(self, map_name: str) -> None:
        if not map_name:
            raise InvalidAvRequest("ScenarioPack map_name is required to generate CARLA world")
        if self._server_version is None and not self._ensure_connected():
            raise AvTimeout("Timed out connecting to CARLA before loading world")
        if self._client is None:
            raise AvUnavailable("CARLA client is not available")

        carla_map_name = None
        opendrive_name = map_name
        opendrive_path = Path(os.path.join(self._xodr_root, f"{opendrive_name}.xodr")).resolve()

        if (
            self._reuse_generated_world
            and self._world is not None
            and self._loaded_map_name == map_name
            and self._loaded_opendrive_path == opendrive_path
        ):
            logger.debug("Reusing generated CARLA world for OpenDRIVE map: %s", opendrive_path)
            self._map = self._world.get_map()
            return

        world = None
        if carla_map_name:
            world = self._client.load_world(carla_map_name, reset_settings=False)
        elif opendrive_path and hasattr(self._client, "generate_opendrive_world"):
            opendrive_path = Path(opendrive_path)
            if not opendrive_path.exists():
                raise InvalidAvRequest("OpenDRIVE path not found for CARLA world generation")

            # read opendrive file
            try:
                with open(opendrive_path, encoding="utf-8") as f:
                    opendrive_str = f.read()
            except OSError as exc:
                raise InvalidAvRequest(
                    f"Failed to read OpenDRIVE map file: {opendrive_path}"
                ) from exc
            # OpenDRIVE world generation can take minutes — bump the
            # client timeout, but guarantee it gets restored even if
            # generation raises (otherwise every subsequent CARLA call
            # on this client inherits the inflated 300s timeout).
            default_timeout = float(os.environ.get("CARLA_TIMEOUT", 10.0))
            self._client.set_timeout(300.0)
            try:
                logger.debug("Generating CARLA world from OpenDRIVE: %s", opendrive_path)
                try:
                    world = self._client.generate_opendrive_world(
                        opendrive_str,
                        carla.OpendriveGenerationParameters(
                            vertex_distance=2.0,
                            max_road_length=3000.0,
                            wall_height=0.0,
                            additional_width=0.6,
                            smooth_junctions=True,
                            enable_mesh_visibility=True,
                        ),
                    )
                except Exception as exc:
                    raise InvalidAvRequest(
                        f"Failed to generate CARLA world from OpenDRIVE: {opendrive_path}"
                    ) from exc
                logger.debug("Generated CARLA world from OpenDRIVE: %s", opendrive_path)
            finally:
                self._client.set_timeout(default_timeout)
        else:
            raise InvalidAvRequest("Cannot determine CARLA world to load")

        if world is None:
            world = self._client.get_world()

        self._world = world
        try:
            self._map = world.get_map()
        except Exception as exc:
            raise InvalidAvRequest("Failed to read CARLA map from generated world") from exc
        self._loaded_map_name = map_name
        self._loaded_opendrive_path = opendrive_path

    def _prepare_reused_server_state(self) -> None:
        if self._client is None:
            return
        previous_world = self._world
        try:
            self._world = self._client.get_world()
            self._map = self._world.get_map()
        except Exception:
            logger.exception("Failed to get CARLA world while preparing reused server")
            return
        if self._world is not previous_world:
            self._loaded_map_name = None
            self._loaded_opendrive_path = None
        self._clear_dynamic_actors()

    def _clear_dynamic_actors(self) -> None:
        clear_dynamic_actors(
            self._world,
            client=self._client,
            traffic_manager_port=getattr(self, "_traffic_manager_port", 8000),
            manage_traffic_manager=getattr(self, "_manage_traffic_manager_sync", False),
            log=logger,
        )

    def _get_spawn_position(
        self,
        init_obs: ObservationData | None,
        sps: ScenarioPackData | None,
    ):
        if init_obs is not None and getattr(init_obs, "ego", None) is not None:
            return init_obs.ego.kinematic
        if sps is None:
            return None
        ego = getattr(sps, "ego", None)
        if ego is None:
            return None
        for attr in ("spawn_config", "spawn"):
            cfg = getattr(ego, attr, None)
            if cfg is None:
                continue
            pos = getattr(cfg, "position", None)
            if pos is not None:
                return pos
        return None

    def _apply_world_settings(self) -> None:
        if self._world is None:
            return
        settings = self._world.get_settings()
        settings.synchronous_mode = self._sync
        logger.debug("Synchronous mode = %s", settings.synchronous_mode)
        settings.no_rendering_mode = self._no_rendering
        logger.debug("No rendering mode = %s", settings.no_rendering_mode)
        if self._fixed_delta_seconds is not None:
            logger.debug("Setting fixed_delta_seconds = %s", self._fixed_delta_seconds)
            settings.fixed_delta_seconds = float(self._fixed_delta_seconds)
        self._world.apply_settings(settings)

    def _find_blueprint(self, bp_lib, candidates: tuple[str, ...]):
        for pattern in candidates:
            try:
                if "*" not in pattern:
                    return bp_lib.find(pattern)
                matches = bp_lib.filter(pattern)
            except Exception:
                logger.debug("CARLA blueprint lookup failed for pattern %s", pattern, exc_info=True)
                continue
            if matches:
                return matches[0]
        return None

    def _pick_blueprint(self, obj_type: RoadObjectType):
        if self._world is None:
            return None
        candidates = BLUEPRINT_CANDIDATES.get(
            obj_type, BLUEPRINT_CANDIDATES[RoadObjectType.UNKNOWN]
        )
        return self._find_blueprint(self._world.get_blueprint_library(), candidates)

    @staticmethod
    def _role_name_for_tracking_id(tracking_id: int) -> str:
        return f"agent_{tracking_id}"[:255]

    def _spawn_actor_allowing_observation_overlap(self, blueprint, transform):
        if self._world is None:
            return None

        actor = self._world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            return actor

        base_loc = transform.location
        rotation = transform.rotation
        retry_offsets = (
            max(float(getattr(self, "_spawn_z_offset", 0.0)), 5.0),
            10.0,
            20.0,
            50.0,
        )
        tried_z = {round(float(base_loc.z), 6)}
        for offset in retry_offsets:
            spawn_z = float(base_loc.z) + offset
            if round(spawn_z, 6) in tried_z:
                continue
            tried_z.add(round(spawn_z, 6))
            elevated_transform = self._carla.Transform(
                self._carla.Location(base_loc.x, base_loc.y, spawn_z),
                rotation,
            )
            actor = self._world.try_spawn_actor(blueprint, elevated_transform)
            if actor is not None:
                return actor
        return None

    def _make_observation_actor_kinematic(self, actor) -> None:
        if actor is None:
            return
        with contextlib.suppress(Exception):
            actor.set_simulate_physics(False)
        with contextlib.suppress(Exception):
            actor.set_enable_gravity(False)

    @staticmethod
    def _rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float):
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )

    @staticmethod
    def _matrix_multiply(left, right):
        return tuple(
            tuple(sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3))
            for row in range(3)
        )

    @staticmethod
    def _matrix_vector(matrix, vector):
        return tuple(sum(matrix[row][k] * vector[k] for k in range(3)) for row in range(3))

    @staticmethod
    def _matrix_transpose(matrix):
        return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))

    @staticmethod
    def _matrix_to_rotation(matrix):
        pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
        if abs(math.cos(pitch)) > 1e-8:
            roll = math.atan2(matrix[2][1], matrix[2][2])
            yaw = math.atan2(matrix[1][0], matrix[0][0])
        else:
            roll = 0.0
            yaw = math.atan2(-matrix[0][1], matrix[1][1])
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    def _object_transform(self, state: ObjectStateData, actor=None, z_offset: float = 0.0):
        kin = state.kinematic
        kin_loc = self._to_carla_location(kin)
        if z_offset:
            kin_loc.z += z_offset
        kin_yaw = self._to_carla_yaw(float(kin.yaw))
        fallback = self._carla.Transform(
            kin_loc,
            self._carla.Rotation(pitch=0.0, yaw=kin_yaw, roll=0.0),
        )

        shape = getattr(state, "shape", None)
        bounding_box = getattr(actor, "bounding_box", None)
        if shape is None or shape.type != ShapeType.BOUNDING_BOX or bounding_box is None:
            return fallback

        center = shape.center
        kin_rotation = self._rotation_matrix(0.0, 0.0, kin_yaw)
        center_rotation = self._rotation_matrix(
            math.degrees(float(center.roll)),
            math.degrees(float(center.pitch)),
            math.degrees(float(center.yaw)) * self._yaw_sign,
        )
        box_world_rotation = self._matrix_multiply(kin_rotation, center_rotation)
        center_offset = (
            float(center.x),
            float(center.y) * self._coordinate_y_sign,
            float(center.z),
        )
        rotated_center = self._matrix_vector(kin_rotation, center_offset)
        box_world_location = (
            float(kin_loc.x) + rotated_center[0],
            float(kin_loc.y) + rotated_center[1],
            float(kin_loc.z) + rotated_center[2],
        )

        local_location = getattr(bounding_box, "location", None)
        local_rotation = getattr(bounding_box, "rotation", None)
        actor_box_rotation = self._rotation_matrix(
            float(getattr(local_rotation, "roll", 0.0)),
            float(getattr(local_rotation, "pitch", 0.0)),
            float(getattr(local_rotation, "yaw", 0.0)),
        )
        actor_rotation = self._matrix_multiply(
            box_world_rotation, self._matrix_transpose(actor_box_rotation)
        )
        actor_box_offset = self._matrix_vector(
            actor_rotation,
            (
                float(getattr(local_location, "x", 0.0)),
                float(getattr(local_location, "y", 0.0)),
                float(getattr(local_location, "z", 0.0)),
            ),
        )
        actor_location = self._carla.Location(
            box_world_location[0] - actor_box_offset[0],
            box_world_location[1] - actor_box_offset[1],
            box_world_location[2] - actor_box_offset[2],
        )
        roll, pitch, yaw = self._matrix_to_rotation(actor_rotation)
        return self._carla.Transform(
            actor_location,
            self._carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
        )

    def _update_and_tick(self, observation: ObservationData) -> None:
        if self._world is None:
            return

        def apply_state(actor, state, *, make_kinematic: bool = False) -> None:
            if actor is None:
                return
            if make_kinematic:
                self._make_observation_actor_kinematic(actor)
            try:
                transform = self._object_transform(state, actor)
            except ValueError as exc:
                raise InvalidAvRequest(str(exc)) from exc
            actor.set_transform(transform)

            kin = state.kinematic
            speed = float(kin.speed)
            yaw_carla_deg = self._to_carla_yaw(float(kin.yaw))
            yaw_carla_rad = math.radians(yaw_carla_deg)
            vx = speed * math.cos(yaw_carla_rad)
            vy = speed * math.sin(yaw_carla_rad)
            vel = self._carla.Vector3D(vx, vy, 0.0)
            try:
                actor.set_target_velocity(vel)
            except Exception:
                with contextlib.suppress(Exception):
                    actor.set_velocity(vel)

            if abs(float(kin.yaw_rate)) > 1e-6:
                ang_z = math.degrees(float(kin.yaw_rate)) * self._yaw_sign
                ang = self._carla.Vector3D(0.0, 0.0, ang_z)
                try:
                    actor.set_target_angular_velocity(ang)
                except Exception:
                    with contextlib.suppress(Exception):
                        actor.set_angular_velocity(ang)

        if self._vehicle is None:
            # Auto-spawn is currently disabled (the `_spawn_ego` call
            # below is intentionally unreachable while the obs-based
            # spawn path is in flight). Surface a clear configuration
            # error rather than continuing with `_vehicle is None`.
            raise RuntimeError(
                "Ego vehicle not found in scenario pack and auto-spawn is disabled. "
                "Define the ego in the scenario pack or enable auto-spawn."
            )

        apply_state(self._vehicle, observation.ego)

        agents = list(observation.agents)
        tracking_ids = [agent.tracking_id for agent in agents]
        use_tracking_ids = all(tracking_id is not None for tracking_id in tracking_ids)
        if use_tracking_ids and len(set(tracking_ids)) != len(tracking_ids):
            raise InvalidAvRequest("Observation contains duplicate agent tracking IDs")

        if not use_tracking_ids:
            self._destroy_other_actors()
            self._using_tracking_ids = False
            for observed_agent in agents:
                obj = observed_agent.state
                bp = self._pick_blueprint(obj.type)
                if bp is None:
                    raise InvalidAvRequest(f"No blueprint for object type {obj.type}")
                if bp.has_attribute("role_name"):
                    bp.set_attribute("role_name", "agent")
                try:
                    transform = self._object_transform(obj, z_offset=self._spawn_z_offset)
                except ValueError as exc:
                    raise InvalidAvRequest(str(exc)) from exc
                actor = self._spawn_actor_allowing_observation_overlap(bp, transform)
                if actor is None:
                    raise AvPreconditionFailed("Failed to spawn stateless observation actor")
                self._spawned_actor_ids.add(actor.id)
                self._other_actors.append(actor)
                self._other_actor_types.append(obj.type)
                apply_state(actor, obj, make_kinematic=True)
        else:
            if not getattr(self, "_using_tracking_ids", False):
                self._destroy_other_actors()
            self._using_tracking_ids = True
            observed_keys = set(tracking_ids)
            for observed_agent in agents:
                key = observed_agent.tracking_id
                obj = observed_agent.state
                actor = self._other_actors_by_key.get(key)
                obj_type = obj.type
                if (
                    actor is None
                    or (hasattr(actor, "is_alive") and not actor.is_alive)
                    or self._other_actor_types_by_key.get(key) != obj_type
                ):
                    if actor is not None:
                        actor_id = getattr(actor, "id", None)
                        if destroy_actor(actor, log=logger, label="actor"):
                            self._spawned_actor_ids.discard(actor_id)
                    bp = self._pick_blueprint(obj_type)
                    if bp is None:
                        raise InvalidAvRequest(f"No blueprint for object type {obj_type}")
                    if bp.has_attribute("role_name"):
                        bp.set_attribute("role_name", self._role_name_for_tracking_id(key))
                    try:
                        transform = self._object_transform(obj, z_offset=self._spawn_z_offset)
                    except ValueError as exc:
                        raise InvalidAvRequest(str(exc)) from exc
                    actor = self._spawn_actor_allowing_observation_overlap(bp, transform)
                    if actor is None:
                        raise AvPreconditionFailed(f"Failed to spawn actor for tracking ID {key}")
                    self._spawned_actor_ids.add(actor.id)
                    self._other_actors_by_key[key] = actor
                    self._other_actor_types_by_key[key] = obj_type

                apply_state(self._other_actors_by_key.get(key), obj, make_kinematic=True)

            for key in set(self._other_actors_by_key) - observed_keys:
                actor = self._other_actors_by_key.pop(key)
                self._other_actor_types_by_key.pop(key, None)
                actor_id = getattr(actor, "id", None)
                if destroy_actor(actor, log=logger, label="stale actor"):
                    self._spawned_actor_ids.discard(actor_id)

            self._other_actors = list(self._other_actors_by_key.values())
            self._other_actor_types = list(self._other_actor_types_by_key.values())

        if self._sync:
            self._world.tick()
        else:
            self._world.wait_for_tick()

    def _destroy_other_actors(self) -> None:
        other_actors_by_key = getattr(self, "_other_actors_by_key", {})
        other_actor_types_by_key = getattr(self, "_other_actor_types_by_key", {})
        actors = list(other_actors_by_key.values()) or list(self._other_actors)
        for actor in actors:
            actor_id = getattr(actor, "id", None)
            if destroy_actor(actor, log=logger, label="actor"):
                self._spawned_actor_ids.discard(actor_id)
        self._other_actors.clear()
        self._other_actor_types.clear()
        other_actors_by_key.clear()
        other_actor_types_by_key.clear()
        self._using_tracking_ids = False

    def _reset_other_actor_state(self) -> None:
        self._other_actors.clear()
        self._other_actor_types.clear()
        self._other_actors_by_key.clear()
        self._other_actor_types_by_key.clear()
        self._spawned_actor_ids.clear()
        self._using_tracking_ids = False

    def _destroy_spawned_actors(self) -> None:
        if self._world is None:
            self._other_actors.clear()
            self._other_actor_types.clear()
            self._other_actors_by_key.clear()
            self._other_actor_types_by_key.clear()
            self._spawned_actor_ids.clear()
            self._using_tracking_ids = False
            return

        force_async_world_for_cleanup(
            self._world,
            client=self._client,
            traffic_manager_port=getattr(self, "_traffic_manager_port", 8000),
            manage_traffic_manager=getattr(self, "_manage_traffic_manager_sync", False),
            log=logger,
        )

        destroyed_actor_ids = set()
        for actor_id in list(self._spawned_actor_ids):
            actor = None
            try:
                if hasattr(self._world, "get_actor"):
                    actor = self._world.get_actor(actor_id)
            except Exception:
                logger.exception("Failed to look up spawned actor %s", actor_id)
            if destroy_actor(actor, log=logger, label="spawned actor"):
                destroyed_actor_ids.add(actor_id)

        other_actors_by_key = getattr(self, "_other_actors_by_key", {})
        other_actor_types_by_key = getattr(self, "_other_actor_types_by_key", {})
        other_actors = list(other_actors_by_key.values()) or list(self._other_actors)
        for actor in [self._vehicle, *other_actors]:
            actor_id = getattr(actor, "id", None)
            if actor_id is not None and actor_id in destroyed_actor_ids:
                continue
            destroy_actor(actor, log=logger, label="actor")

        self._vehicle = None
        for actor in other_actors:
            actor_id = getattr(actor, "id", None)
            if actor_id is not None:
                self._spawned_actor_ids.discard(actor_id)
        self._other_actors.clear()
        self._other_actor_types.clear()
        other_actors_by_key.clear()
        other_actor_types_by_key.clear()
        self._spawned_actor_ids.clear()
        self._using_tracking_ids = False

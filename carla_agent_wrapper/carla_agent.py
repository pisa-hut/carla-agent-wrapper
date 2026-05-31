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
    ControlCommand,
    ControlMode,
    InitRequest,
    InitResponse,
    ObjectStateData,
    ResetRequest,
    ResetResponse,
    RoadObjectType,
    ScenarioPackData,
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

OBJECT_IDENTITY_ATTRS = ("id", "object_id", "track_id", "external_id", "name")
OBJECT_IDENTITY_MODES = {"index", "provided", "stateless"}

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

    def _connect(self, timeout: float = 2.0):
        if self._server_version is not None:
            return
        logger.info("Connecting to CARLA...")
        self._client = carla.Client(
            os.environ.get("CARLA_HOST", "localhost"),
            int(os.environ.get("CARLA_PORT", 2000)),
        )
        try:
            self._client.set_timeout(timeout)
            self._server_version = self._client.get_server_version()
        finally:
            self._client.set_timeout(float(os.environ.get("CARLA_TIMEOUT", 10.0)))
        logger.info("Connected to CARLA")

    def _ensure_connected(self) -> bool:
        config = getattr(self, "config", {}) or {}
        timeout = float(
            config.get(
                "carla_connect_timeout_seconds", float(config.get("max_retry_times", 15)) * 2
            )
        )
        retry_interval = float(config.get("retry_interval_seconds", 2.0))
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
            return float(pos.yaw)
        world = getattr(pos, "world", None)
        if world is not None and hasattr(world, "h"):
            return float(world.h)
        if hasattr(pos, "h"):
            return float(pos.h)
        return 0.0

    def _to_carla_location(self, pos) -> Any:
        try:
            x, y, z = self._extract_xyz(pos)
        except Exception as exc:
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
        speed = float(speed)
        if self._target_speed_is_mps:
            speed = speed * 3.6
        return speed

    def _validate_config(self) -> None:
        if self._agent_type not in VALID_AGENT_TYPES:
            raise ValueError(f"Unsupported CARLA agent_type: {self._agent_type!r}")
        if self._agent_type == "behavior" and self._behavior not in VALID_BEHAVIORS:
            raise ValueError(
                f"Unsupported CARLA behavior: {self._behavior!r}. "
                f"Expected one of: {', '.join(sorted(VALID_BEHAVIORS))}"
            )

    def _config_sign(self, name: str, default: float) -> float:
        value = float(self.config.get(name, default))
        if abs(value) < 1e-6:
            raise ValueError(f"{name} must be non-zero")
        return 1.0 if value > 0 else -1.0

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
            print(
                f"Initialized CARLA BehaviorAgent with behavior={self._behavior}, target_speed_kmh={target_speed_kmh}"
            )

        elif self._agent_type == "basic":
            agent = self._BasicAgent(
                self._vehicle,
                target_speed=target_speed_kmh,
                opt_dict=agent_opt_dict,
                map_inst=self._map,
            )
            self._configure_local_planner(agent)
            print(f"Initialized CARLA BasicAgent with target_speed_kmh={target_speed_kmh}")
        elif self._agent_type in ("constant_velocity", "constant-velocity"):
            agent = self._ConstantVelocityAgent(
                self._vehicle,
                target_speed=target_speed_kmh,
                opt_dict=agent_opt_dict,
                map_inst=self._map,
            )
        else:
            raise ValueError(f"Unsupported CARLA agent_type: {self._agent_type!r}")

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

    def init(self, request: InitRequest) -> InitResponse | None:
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
        self._local_planner_base_min_distance = float(
            self.config.get("local_planner_base_min_distance", 1.0)
        )
        self._local_planner_distance_ratio = float(
            self.config.get("local_planner_distance_ratio", 0.0)
        )
        self._route_sampling_resolution = float(self.config.get("route_sampling_resolution", 0.5))

        legacy_yaw_sign = self._config_sign("yaw_sign", -1.0)
        self._coordinate_y_sign = self._config_sign("coordinate_y_sign", legacy_yaw_sign)
        self._yaw_sign = self._config_sign("yaw_sign", legacy_yaw_sign)
        self._steer_sign = self._config_sign("steer_sign", legacy_yaw_sign)
        self._yaw_offset_deg = float(self.config.get("yaw_offset_deg", 0.0))
        self._spawn_z_offset = float(self.config.get("spawn_z_offset", 3.0))
        self._xodr_root = Path(self.config.get("xodr_root", "/mnt/map/xodr"))
        self._reuse_generated_world = bool(self.config.get("reuse_generated_world", True))
        self._traffic_manager_port = int(os.environ.get("CARLA_TM_PORT", 8000))
        self._object_identity_mode = str(
            self.config.get("object_identity_mode", "stateless")
        ).lower()
        if self._object_identity_mode not in OBJECT_IDENTITY_MODES:
            raise ValueError(
                f"Unsupported object_identity_mode: {self._object_identity_mode!r}. "
                f"Expected one of: {', '.join(sorted(OBJECT_IDENTITY_MODES))}"
            )

        if not self._ensure_connected():
            return InitResponse(success=False, msg="Failed to connect to CARLA within timeout")

        self._prepare_reused_server_state()
        self._quit_flag = False
        return InitResponse(success=True, msg="CARLA agent initialized successfully")

    def reset(
        self,
        request: ResetRequest,
    ) -> ControlCommand | ResetResponse:
        if not self._finalized:
            self._finalize()

        self._finalized = False
        self._output_dir = request.output_dir
        # os.makedirs(self._output_dir, exist_ok=True)
        sps = request.scenario_pack
        init_obs = request.initial_observation
        self._sps = sps
        self._quit_flag = False

        try:
            if sps is None:
                raise RuntimeError("ScenarioPack is required to prepare CARLA agent")
            self._ensure_world(sps.map_name)
            self._clear_dynamic_actors()
            self._vehicle = None
            logger.info("Ego vehicle found: %s", self._vehicle)
            if self._vehicle is None:
                self._vehicle = self._spawn_ego(init_obs, sps)

            self._apply_world_settings()

            target_speed_kmh = self._get_target_speed_kmh(sps)
            self._agent = self._build_agent(target_speed_kmh)
            if self._agent is None:
                raise RuntimeError("Failed to create CARLA agent")

            if self._random_destination:
                if self._map is None:
                    raise RuntimeError("CARLA map not available for destination picking")
                spawn_points = self._map.get_spawn_points()
                if not spawn_points:
                    raise RuntimeError("No spawn points available for random destination")
                dest = random.choice(spawn_points).location
            else:
                goal_pos = sps.ego.goal_config.position if sps is not None else None
                if goal_pos is None:
                    if self._map is None:
                        raise RuntimeError("Goal position missing and CARLA map not available")
                    logger.warning(
                        "Goal position missing; using random destination from spawn points."
                    )
                    spawn_points = self._map.get_spawn_points()
                    if not spawn_points:
                        raise RuntimeError("No spawn points available for destination fallback")
                    dest = random.choice(spawn_points).location
                else:
                    dest = self._to_carla_location(goal_pos)
            dest.z += self._spawn_z_offset  # to avoid underground issues

            start_transform = self._vehicle.get_transform()
            start_wp = self._snap_to_waypoint(start_transform.location, "ego start")
            end_wp = self._snap_to_waypoint(dest, "destination")
            logger.info(
                "Route start: %s, yaw: %.3f, snapped to: %s",
                start_transform.location,
                self._from_carla_yaw(start_transform.rotation.yaw),
                start_wp.transform.location,
            )
            logger.info(
                "Route destination: %s, snapped to: %s",
                dest,
                end_wp.transform.location,
            )
            try:
                self._agent.set_destination(end_wp.transform.location, start_wp.transform.location)
            except Exception as exc:
                raise RuntimeError(
                    f"CARLA agent_type={self._agent_type!r} failed to plan a route. "
                    f"start={self._format_waypoint(start_wp)}, "
                    f"destination={self._format_waypoint(end_wp)}"
                ) from exc

            return self.step(StepRequest(observation=init_obs if init_obs is not None else []))
        except Exception:
            logger.exception("Failed to reset CARLA agent; finalizing partial state")
            self._finalize()
            raise

    def step(self, request: StepRequest) -> ControlCommand | StepResponse:
        obs = request.observation
        self._update_and_tick(obs)
        if self._agent is None:
            raise RuntimeError("CARLA agent is not initialized")

        control = self._agent.run_step()
        if hasattr(self._agent, "done") and self._agent.done():
            self._quit_flag = True

        steer_sv = float(control.steer) / getattr(self, "_steer_sign", 1.0)
        return ControlCommand(
            mode=ControlMode.THROTTLE_STEER_BREAK,
            payload={
                "throttle": float(control.throttle),
                "brake": float(control.brake),
                "steer": steer_sv,
            },
        )

    def stop(self) -> None:
        self._finalize()
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

    def should_quit(self) -> bool:
        return self._quit_flag

    def _spawn_ego(self, init_obs: list[ObjectStateData] | None, sps: ScenarioPackData):
        if self._world is None:
            raise RuntimeError("CARLA world not available")

        bp_lib = self._world.get_blueprint_library()
        ego_bp = self._find_blueprint(bp_lib, (self._ego_bp_id, "vehicle.*"))
        if ego_bp is None:
            raise RuntimeError("No vehicle blueprints available in CARLA")

        if ego_bp.has_attribute("role_name"):
            ego_bp.set_attribute("role_name", self._ego_role_name)

        pos = self._get_spawn_position(init_obs, sps)
        if pos is None:
            raise RuntimeError("No spawn position available for ego vehicle")
        carla_pos = self._to_carla_location(pos)
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
                raise RuntimeError("Failed to spawn ego vehicle (no spawn points)")
            ego = self._world.try_spawn_actor(ego_bp, spawn_points[0])
            if ego is None:
                raise RuntimeError("Failed to spawn ego vehicle")
        self._spawned_actor_ids.add(ego.id)

        try:
            phys = ego.get_physics_control()
            max_steer = max([w.max_steer_angle for w in phys.wheels])
            self._max_steer_rad = math.radians(max_steer)
        except Exception:
            self._max_steer_rad = None

        logger.info("Ego vehicle spawned at %s with yaw %.3f", carla_pos, self._extract_yaw(pos))
        return ego

    def _to_carla_yaw(self, yaw_rad: float) -> float:
        return self._yaw_sign * math.degrees(yaw_rad) + self._yaw_offset_deg

    def _from_carla_yaw(self, yaw_deg: float) -> float:
        return math.radians((yaw_deg - self._yaw_offset_deg) * self._yaw_sign)

    def _snap_to_waypoint(self, location, label: str):
        if self._map is None:
            raise RuntimeError("CARLA map not available for route planning")
        waypoint = self._map.get_waypoint(location, project_to_road=True)
        if waypoint is None:
            raise RuntimeError(f"Failed to project {label} location onto CARLA map: {location}")
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
            raise RuntimeError("ScenarioPack map_name is required to generate CARLA world")
        if self._server_version is None and not self._ensure_connected():
            raise RuntimeError("Failed to connect to CARLA before loading world")
        if self._client is None:
            raise RuntimeError("CARLA client is not available")

        carla_map_name = None
        opendrive_name = map_name
        opendrive_path = Path(os.path.join(self._xodr_root, f"{opendrive_name}.xodr")).resolve()

        if (
            self._reuse_generated_world
            and self._world is not None
            and self._loaded_map_name == map_name
            and self._loaded_opendrive_path == opendrive_path
        ):
            logger.info("Reusing generated CARLA world for OpenDRIVE map: %s", opendrive_path)
            self._map = self._world.get_map()
            return

        world = None
        if carla_map_name:
            world = self._client.load_world(carla_map_name, reset_settings=False)
        elif opendrive_path and hasattr(self._client, "generate_opendrive_world"):
            opendrive_path = Path(opendrive_path)
            if not opendrive_path.exists():
                raise RuntimeError("OpenDRIVE path not found for CARLA world generation")

            # read opendrive file
            with open(opendrive_path, encoding="utf-8") as f:
                opendrive_str = f.read()
            # OpenDRIVE world generation can take minutes — bump the
            # client timeout, but guarantee it gets restored even if
            # generation raises (otherwise every subsequent CARLA call
            # on this client inherits the inflated 300s timeout).
            default_timeout = float(os.environ.get("CARLA_TIMEOUT", 10.0))
            self._client.set_timeout(300.0)
            try:
                logger.info("Generating CARLA world from OpenDRIVE: %s", opendrive_path)
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
                logger.info("Generated CARLA world from OpenDRIVE: %s", opendrive_path)
            finally:
                self._client.set_timeout(default_timeout)
        else:
            raise RuntimeError("Cannot determine CARLA world to load")

        if world is None:
            world = self._client.get_world()

        self._world = world
        self._map = world.get_map()
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
            log=logger,
        )

    def _get_spawn_position(
        self,
        init_obs: list[ObjectStateData] | None,
        sps: ScenarioPackData | None,
    ):
        if init_obs:
            try:
                return init_obs[0].kinematic
            except Exception:
                pass
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
        logger.info("Synchronous mode = %s", settings.synchronous_mode)
        settings.no_rendering_mode = self._no_rendering
        logger.info("No rendering mode = %s", settings.no_rendering_mode)
        if self._fixed_delta_seconds is not None:
            logger.info("Setting fixed_delta_seconds = %s", self._fixed_delta_seconds)
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

    def _provided_object_identity(self, obj: ObjectStateData):
        for attr in OBJECT_IDENTITY_ATTRS:
            value = getattr(obj, attr, None)
            if value not in (None, ""):
                return attr, value
        return None

    def _object_identity(self, obj: ObjectStateData, index: int):
        if self._object_identity_mode == "index":
            return "index", index
        if self._object_identity_mode == "stateless":
            return "frame", index

        identity = self._provided_object_identity(obj)
        if identity is None:
            raise RuntimeError(
                "object_identity_mode='provided' requires each non-ego object to expose one of: "
                f"{', '.join(OBJECT_IDENTITY_ATTRS)}"
            )
        return identity

    def _role_name_for_object_key(self, key) -> str:
        raw = "_".join(str(part) for part in key)
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in raw)
        return f"agent_{safe}"[:255]

    def _update_and_tick(self, obs: list[ObjectStateData]) -> None:
        if self._world is None:
            return

        def make_transform(kin, z_offset: float = 0.0):
            loc = self._to_carla_location(kin)
            if z_offset:
                loc.z += z_offset
            rot = self._carla.Rotation(
                pitch=0.0,
                yaw=self._to_carla_yaw(float(kin.yaw)),
                roll=0.0,
            )
            return self._carla.Transform(loc, rot)

        def apply_kinematic(actor, kin) -> None:
            if actor is None:
                return
            try:
                actor.set_transform(make_transform(kin))
            except Exception:
                logger.exception("Failed to set actor transform")

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

        if not obs:
            if self._sync:
                self._world.tick()
            else:
                self._world.wait_for_tick()
            return

        if self._vehicle is None:
            # Auto-spawn is currently disabled (the `_spawn_ego` call
            # below is intentionally unreachable while the obs-based
            # spawn path is in flight). Surface a clear configuration
            # error rather than continuing with `_vehicle is None`.
            raise RuntimeError(
                "Ego vehicle not found in scenario pack and auto-spawn is disabled. "
                "Define the ego in the scenario pack or enable auto-spawn."
            )

        ego_state = obs[0].kinematic
        apply_kinematic(self._vehicle, ego_state)

        if getattr(self, "_object_identity_mode", "stateless") == "stateless":
            self._destroy_other_actors()

        observed_keys = set()
        for idx, obj in enumerate(obs[1:]):
            key = self._object_identity(obj, idx)
            observed_keys.add(key)
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
                    logger.warning("No blueprint for object type %s", obj_type)
                    self._other_actors_by_key[key] = None
                    self._other_actor_types_by_key[key] = obj_type
                    continue
                if bp.has_attribute("role_name"):
                    bp.set_attribute("role_name", self._role_name_for_object_key(key))
                transform = make_transform(obj.kinematic, z_offset=self._spawn_z_offset)
                actor = self._world.try_spawn_actor(bp, transform)
                if actor is None:
                    logger.warning("Failed to spawn actor for object %s", key)
                else:
                    self._spawned_actor_ids.add(actor.id)
                self._other_actors_by_key[key] = actor
                self._other_actor_types_by_key[key] = obj_type

            apply_kinematic(self._other_actors_by_key.get(key), obj.kinematic)

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

    def _destroy_spawned_actors(self) -> None:
        if self._world is None:
            self._other_actors.clear()
            self._other_actor_types.clear()
            self._other_actors_by_key.clear()
            self._other_actor_types_by_key.clear()
            self._spawned_actor_ids.clear()
            return

        force_async_world_for_cleanup(
            self._world,
            client=self._client,
            traffic_manager_port=getattr(self, "_traffic_manager_port", 8000),
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

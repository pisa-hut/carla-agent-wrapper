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
from pisa_api.control_pb2 import CtrlCmd, CtrlMode
from pisa_api.object_pb2 import ObjectState, RoadObjectType
from pisa_api.scenario_pb2 import ScenarioPack

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


class CarlaAgentAV:
    """
    CARLA automatic-control style AV adapter.
    - init(): connect to CARLA, prepare agent classes
    - reset(): find ego by role_name, create agent, set destination/speed
    - step(): run agent.run_step(), convert to CtrlCmd
    """

    def __init__(self):
        self._carla = carla
        self._BehaviorAgent = BehaviorAgent
        self._BasicAgent = BasicAgent
        self._ConstantVelocityAgent = ConstantVelocityAgent

        self._host = os.environ.get("CARLA_HOST", "localhost")
        self._port = int(os.environ.get("CARLA_PORT", 2000))
        self._timeout = float(os.environ.get("CARLA_TIMEOUT", 10.0))

        self._original_settings = None
        self._spawned_actor_ids = set()

        self._client = None
        self._server_version = None

        self._quit_flag = False

        self._server_log_path = "/mnt/output/carla_server"
        os.makedirs(self._server_log_path, exist_ok=True)
        # File handles are intentionally leaked here — they need to outlive
        # this scope so the long-running CARLA subprocess can keep writing.
        # SIM115 wants `with open(...)` which would close the file before
        # the subprocess emits anything.
        subprocess.Popen(  # noqa: SIM115
            ["/app/carla_server.sh"],
            stdout=open(f"{self._server_log_path}/stdout.log", "w"),  # noqa: SIM115
            stderr=open(f"{self._server_log_path}/stderr.log", "w"),  # noqa: SIM115
        )

        while self._server_version is None:
            try:
                self._connect()
            except Exception:
                logger.exception("Failed to connect to CARLA, retrying in 2 seconds...")
                time.sleep(2)
                continue
            break
        time.sleep(2)  # wait a bit for the server to be fully ready
        print("CARLA service initialized")

        # reset
        self._world = None
        self._map = None
        self._vehicle = None
        self._agent = None
        self._other_actors: list[Any] = []
        self._other_actor_types: list[RoadObjectType] = []

    def _connect(self):
        if self._server_version is not None:
            return
        print("Connecting to CARLA...")
        self._client = carla.Client(
            os.environ.get("CARLA_HOST", "localhost"),
            int(os.environ.get("CARLA_PORT", 2000)),
        )
        self._client.set_timeout(2)
        self._server_version = self._client.get_server_version()
        self._client.set_timeout(float(os.environ.get("CARLA_TIMEOUT", 10.0)))
        print("Connected to CARLA")

    def _find_ego_vehicle_once(self):
        if self._world is None:
            return None
        actors = self._world.get_actors().filter("vehicle.*")
        for actor in actors:
            role = actor.attributes.get("role_name", "")
            if role == self._ego_role_name:
                return actor
        return None

    def _wait_for_ego(self, timeout_s: float):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            logger.info("Searching for ego vehicle with role_name=%r...", self._ego_role_name)
            actor = self._find_ego_vehicle_once()
            if actor is not None:
                return actor
            time.sleep(0.05)
        return None

    def _extract_xyz(self, pos) -> tuple[float, float, float]:
        if pos is None:
            return 0.0, 0.0, 0.0
        world = getattr(pos, "world", None)
        if world is not None and hasattr(world, "x"):
            return float(world.x), float(world.y), float(world.z)
        return (
            float(getattr(pos, "x", 0.0)),
            float(getattr(pos, "y", 0.0)),
            float(getattr(pos, "z", 0.0)),
        )

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
        x, y, z = self._extract_xyz(pos)
        y = float(y) * self._yaw_sign
        return self._carla.Location(
            x=float(x),
            y=y,
            z=float(z),
        )

    def _get_target_speed_kmh(self, sps: ScenarioPack) -> float:
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

    def _build_agent(self, target_speed_kmh: float):
        if self._vehicle is None:
            return None

        if self._agent_type == "behavior":
            agent = self._BehaviorAgent(self._vehicle, behavior=self._behavior, map_inst=self._map)
        elif self._agent_type == "basic":
            agent = self._BasicAgent(self._vehicle, map_inst=self._map)
        elif self._agent_type in ("constant_velocity", "constant-velocity"):
            agent = self._ConstantVelocityAgent(
                self._vehicle, target_speed=target_speed_kmh, map_inst=self._map
            )
        else:
            raise ValueError(f"Unsupported agent_type: {self._agent_type}")

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

    def init(self, cfg: dict, output_dir: str, map_name: str) -> None:
        self._output_dir = Path(output_dir)
        self.config = cfg or {}

        self._sync = bool(self.config.get("sync", True))
        self._no_rendering = bool(self.config.get("no_rendering", True))
        self._fixed_delta_seconds = self.config.get("fixed_delta_seconds", 0.05)

        self._ego_role_name = self.config.get("ego_role_name", "hero")
        self._ego_bp_id = self.config.get("ego_bp_id", "vehicle.tesla.model3")
        self._agent_type = str(self.config.get("agent_type", "behavior")).lower()
        self._behavior = str(self.config.get("behavior", "normal")).lower()
        self._random_destination = bool(self.config.get("random_destination", False))
        self._follow_speed_limits = bool(self.config.get("follow_speed_limits", False))
        self._ignore_traffic_lights = bool(self.config.get("ignore_traffic_lights", False))
        self._ignore_stop_signs = bool(self.config.get("ignore_stop_signs", False))
        self._ignore_vehicles = bool(self.config.get("ignore_vehicles", False))

        self._target_speed = self.config.get("target_speed", None)
        self._target_speed_is_mps = bool(self.config.get("target_speed_is_mps", False))

        self._yaw_sign = float(self.config.get("yaw_sign", -1.0))
        self._yaw_offset_deg = float(self.config.get("yaw_offset_deg", 0.0))
        self._spawn_z_offset = float(self.config.get("spawn_z_offset", 3.0))
        self._xodr_root = Path(self.config.get("xodr_root", "/mnt/map/xodr"))
        self._max_retry_times = int(self.config.get("max_retry_times", 10))

    def reset(
        self,
        output_dir: Path,
        sps: ScenarioPack,
        init_obs: list[ObjectState] | None = None,
    ) -> CtrlCmd:
        self._output_dir = Path(output_dir)
        # os.makedirs(self._output_dir, exist_ok=True)
        self._sps = sps
        self._quit_flag = False

        self._ensure_world(sps.map_name)
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
                logger.warning("Goal position missing; using random destination from spawn points.")
                spawn_points = self._map.get_spawn_points()
                if not spawn_points:
                    raise RuntimeError("No spawn points available for destination fallback")
                dest = random.choice(spawn_points).location
            else:
                dest = self._to_carla_location(goal_pos)
        dest.z += self._spawn_z_offset  # to avoid underground issues

        end_wp = self._map.get_waypoint(dest, project_to_road=True)
        print(
            "Start:",
            self._vehicle.get_transform().location,
            "Yaw:",
            self._from_carla_yaw(self._vehicle.get_transform().rotation.yaw),
        )
        print("Destination:", dest, "Snapped to:", end_wp.transform.location)
        print("==============================================")
        self._agent.set_destination(end_wp.transform.location)

        return self.step(
            obs=init_obs if init_obs is not None else [],
            time_stamp_ns=0,
        )

    def step(self, obs: list[ObjectState], time_stamp_ns: int) -> CtrlCmd:
        self._update_and_tick(obs)
        control = self._agent.run_step()
        if hasattr(self._agent, "done") and self._agent.done():
            self._quit_flag = True

        yaw_sign = self._yaw_sign if abs(self._yaw_sign) > 1e-6 else 1.0
        steer_sv = float(control.steer) / yaw_sign

        return CtrlCmd(
            mode=CtrlMode.THROTTLE_STEER_BREAK,
            payload={
                "throttle": float(control.throttle),
                "brake": float(control.brake),
                "steer": steer_sv,
            },
        )

    def stop(self) -> None:
        if self._world is None:
            return
        try:
            self._destroy_spawned_actors()
        except Exception:
            logger.exception("Failed to destroy spawned actors")

        if self._original_settings is not None:
            try:
                self._world.apply_settings(self._original_settings)
                logger.info("Restored original CARLA world settings.")
            except Exception as e:
                logger.warning(f"Failed to restore CARLA world settings: {e}")
        self._agent = None
        self._vehicle = None
        self._world = None
        self._map = None
        self._quit_flag = True

    def should_quit(self) -> bool:
        return self._quit_flag

    def _spawn_ego(self, init_obs: list[ObjectState] | None, sps: ScenarioPack):
        if self._world is None:
            raise RuntimeError("CARLA world not available")

        bp_lib = self._world.get_blueprint_library()
        try:
            ego_bp = bp_lib.find(self._ego_bp_id)
        except Exception as exc:
            candidates = bp_lib.filter("vehicle.*")
            if not candidates:
                raise RuntimeError("No vehicle blueprints available in CARLA") from exc
            ego_bp = candidates[0]

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

    def _ensure_world(self, map_name: str) -> None:
        if self._server_version is None:
            self._connect()

        carla_map_name = None
        opendrive_name = map_name
        opendrive_path = Path(os.path.join(self._xodr_root, f"{opendrive_name}.xodr")).resolve()

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
            self._client.set_timeout(300.0)
            logger.info("Generating CARLA world from OpenDRIVE: %s", opendrive_path)
            world = self._client.generate_opendrive_world(
                opendrive_str,
                carla.OpendriveGenerationParameters(
                    vertex_distance=2.0,
                    max_road_length=500.0,
                    wall_height=0.0,
                    additional_width=0.6,
                    smooth_junctions=True,
                    enable_mesh_visibility=True,
                ),
            )
            self._client.set_timeout(float(os.environ.get("CARLA_TIMEOUT", 10.0)))
            logger.info("Generated CARLA world from OpenDRIVE: %s", opendrive_path)
        else:
            raise RuntimeError("Cannot determine CARLA world to load")

        if world is None:
            world = self._client.get_world()

        self._world = world
        self._map = world.get_map()
        if self._original_settings is None:
            self._original_settings = world.get_settings()

    def _get_spawn_position(self, init_obs: list[ObjectState] | None, sps: ScenarioPack | None):
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

    def _update_and_tick(self, obs: list[ObjectState]) -> None:
        if self._world is None:
            return

        def pick_blueprint(obj_type: RoadObjectType):
            if self._world is None:
                return None
            bp_lib = self._world.get_blueprint_library()
            if obj_type == RoadObjectType.PEDESTRIAN:
                return bp_lib.find("walker.pedestrian.0001")
            elif obj_type == RoadObjectType.BUS:
                return bp_lib.find("vehicle.mitsubishi.fusorosa")
            elif obj_type == RoadObjectType.TRUCK:
                return bp_lib.find("vehicle.carlamotors.carlacola")
            elif obj_type == RoadObjectType.TRAILER:
                return bp_lib.find("vehicle.carlamotors.firetruck")
            elif obj_type == RoadObjectType.VAN:
                return bp_lib.find("vehicle.mercedes.sprinter")
            elif obj_type == RoadObjectType.MOTORCYCLE:
                return bp_lib.find("vehicle.vespa.zx125")
            elif obj_type == RoadObjectType.BICYCLE:
                return bp_lib.find("vehicle.bh.crossbike")
            else:
                candidates = bp_lib.filter("vehicle.*")

            if not candidates and obj_type != RoadObjectType.PEDESTRIAN:
                candidates = bp_lib.filter("vehicle.*")
            if not candidates:
                return None
            return candidates[0]

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
            raise RuntimeError(
                "Removing sps parameter from _spawn_ego, using init_obs instead. If you see this error, it means the ego vehicle was not found and auto-spawn is disabled. Please check your configuration and ensure that the ego vehicle is properly defined in the scenario pack or that auto-spawn is enabled."
            )
            self._vehicle = self._spawn_ego(obs, self._sps)

        ego_state = obs[0].kinematic
        apply_kinematic(self._vehicle, ego_state)

        desired_count = max(len(obs) - 1, 0)
        while len(self._other_actors) < desired_count:
            self._other_actors.append(None)
            self._other_actor_types.append(RoadObjectType.UNKNOWN)
        while len(self._other_actors) > desired_count:
            actor = self._other_actors.pop()
            self._other_actor_types.pop()
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    logger.exception("Failed to destroy extra actor")

        for idx, obj in enumerate(obs[1:]):
            actor = self._other_actors[idx]
            obj_type = obj.type
            if (
                actor is None
                or (hasattr(actor, "is_alive") and not actor.is_alive)
                or self._other_actor_types[idx] != obj_type
            ):
                if actor is not None:
                    try:
                        actor.destroy()
                    except Exception:
                        logger.exception("Failed to destroy actor %s", idx)
                bp = pick_blueprint(obj_type)
                if bp is None:
                    logger.warning("No blueprint for object type %s", obj_type)
                    self._other_actors[idx] = None
                    self._other_actor_types[idx] = obj_type
                    continue
                if bp.has_attribute("role_name"):
                    bp.set_attribute("role_name", f"agent_{idx}")
                transform = make_transform(obj.kinematic, z_offset=self._spawn_z_offset)
                actor = self._world.try_spawn_actor(bp, transform)
                if actor is None:
                    logger.warning("Failed to spawn actor for index %s", idx)
                self._other_actors[idx] = actor
                self._other_actor_types[idx] = obj_type

            apply_kinematic(self._other_actors[idx], obj.kinematic)

        if self._sync:
            self._world.tick()
        else:
            self._world.wait_for_tick()

    def _destroy_spawned_actors(self) -> None:
        if self._world is None:
            return

        if self._vehicle is not None:
            try:
                self._vehicle.destroy()
            except Exception:
                logger.exception("Failed to destroy ego vehicle")
            self._vehicle = None

        if not self._other_actors:
            return

        for actor in list(self._other_actors):
            try:
                actor.destroy()
            except Exception:
                logger.exception("Failed to destroy actor %s", actor.id)

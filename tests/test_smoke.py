import importlib
import math
import random
import sys
from types import ModuleType, SimpleNamespace

import pytest
from pisa_api.av import (
    ObjectKinematicData,
    ObjectStateData,
    ObservationData,
    ObservedAgentData,
    ShapeCenterPoseData,
    ShapeData,
    ShapeDimensionData,
    ShapeType,
)


class _FakeTrafficManager:
    def __init__(self):
        self.sync_calls = []

    def set_synchronous_mode(self, enabled):
        self.sync_calls.append(enabled)


class _FakeClient:
    def __init__(self, world=None, traffic_manager=None):
        self.world = world
        self.traffic_manager = traffic_manager or _FakeTrafficManager()
        self.generated = False

    def get_world(self):
        return self.world

    def get_trafficmanager(self, port):
        return self.traffic_manager

    def generate_opendrive_world(self, *args, **kwargs):
        self.generated = True
        raise AssertionError("generate_opendrive_world should not be called")


class _FakeWorld:
    def __init__(self, actors=None, synchronous_mode=False, fixed_delta_seconds=None):
        self.actors = list(actors or [])
        self.settings = SimpleNamespace(
            synchronous_mode=synchronous_mode,
            fixed_delta_seconds=fixed_delta_seconds,
            no_rendering_mode=False,
        )
        self.applied_settings = None

    def get_actors(self):
        return list(self.actors)

    def get_settings(self):
        return self.settings

    def apply_settings(self, settings):
        self.applied_settings = settings

    def get_map(self):
        return object()

    def get_actor(self, actor_id):
        for actor in self.actors:
            if actor.id == actor_id:
                return actor
        return None


class _FakeBlueprint:
    def __init__(self, blueprint_id="vehicle.test", dimensions=(4.0, 2.0, 1.5)):
        self.id = blueprint_id
        self.dimensions = dimensions
        self.attributes = {}

    def has_attribute(self, name):
        return name == "role_name"

    def set_attribute(self, name, value):
        self.attributes[name] = value


class _FakeBlueprintLibrary:
    def __init__(self, *, find_results=None, filter_results=None):
        self.find_results = dict(find_results or {})
        self.filter_results = dict(filter_results or {})
        self.find_calls = []
        self.filter_calls = []

    def find(self, pattern):
        self.find_calls.append(pattern)
        result = self.find_results.get(pattern)
        if result is None:
            raise KeyError(pattern)
        return result

    def filter(self, pattern):
        self.filter_calls.append(pattern)
        return list(self.filter_results.get(pattern, []))


class _FakeActorWorld(_FakeWorld):
    def __init__(self, blueprints):
        super().__init__([])
        self.blueprints = blueprints
        self.spawned = []
        self.probes = []
        self.tick_calls = 0

    def get_blueprint_library(self):
        return self.blueprints

    def try_spawn_actor(self, blueprint, transform):
        actor = _FakeMovingActor(
            actor_id=len(self.actors) + 10,
            dimensions=getattr(blueprint, "dimensions", (4.0, 2.0, 1.5)),
        )
        if transform.location.z == 10_000.0:
            self.probes.append((actor, blueprint, transform))
        else:
            self.spawned.append((actor, blueprint, transform))
        self.actors.append(actor)
        return actor

    def tick(self):
        self.tick_calls += 1


class _FailFirstSpawnWorld(_FakeActorWorld):
    def __init__(self, blueprints):
        super().__init__(blueprints)
        self.spawn_attempts = []

    def try_spawn_actor(self, blueprint, transform):
        if transform.location.z == 10_000.0:
            return super().try_spawn_actor(blueprint, transform)
        self.spawn_attempts.append(transform)
        if len(self.spawn_attempts) == 1:
            return None
        return super().try_spawn_actor(blueprint, transform)


class _FakeActor:
    def __init__(self, actor_id, type_id="vehicle.test"):
        self.id = actor_id
        self.type_id = type_id
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1


class _DestroyFalseActor(_FakeActor):
    def destroy(self):
        self.destroy_calls += 1
        return False


class _FakeMovingActor(_FakeActor):
    def __init__(self, actor_id, type_id="vehicle.test", dimensions=(2.0, 2.0, 2.0)):
        super().__init__(actor_id, type_id)
        self.transforms = []
        self.simulate_physics_calls = []
        self.enable_gravity_calls = []
        self.bounding_box = SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=0.0),
            extent=SimpleNamespace(
                x=dimensions[0] / 2.0,
                y=dimensions[1] / 2.0,
                z=dimensions[2] / 2.0,
            ),
        )

    def set_transform(self, transform):
        self.transforms.append(transform)

    def set_simulate_physics(self, enabled):
        self.simulate_physics_calls.append(enabled)

    def set_enable_gravity(self, enabled):
        self.enable_gravity_calls.append(enabled)

    def set_target_velocity(self, velocity):
        self.velocity = velocity

    def set_target_angular_velocity(self, velocity):
        self.angular_velocity = velocity


class _FakeAgent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._local_planner = SimpleNamespace()

    def set_target_speed(self, speed):
        self.target_speed = speed

    def _vehicle_obstacle_detected(self, actors, max_distance):
        return ("native", actors, max_distance)

    def set_destination(self, *args, **kwargs):
        pass

    def run_step(self):
        return SimpleNamespace(throttle=0.0, brake=0.0, steer=0.0)


class _FakeAgentWithoutLocalPlanner:
    def __init__(self, *args, **kwargs):
        pass

    def set_target_speed(self, speed):
        self.target_speed = speed

    def set_destination(self, *args, **kwargs):
        pass

    def run_step(self):
        return SimpleNamespace(throttle=0.0, brake=0.0, steer=0.0)


class _RouteFailingAgent(_FakeAgent):
    def set_destination(self, *args, **kwargs):
        raise RuntimeError("route impossible")


class _DoneAgent(_FakeAgent):
    def done(self):
        return True


@pytest.fixture
def carla_agent_module(monkeypatch):
    fake_carla = ModuleType("carla")
    fake_carla.Client = lambda host, port: None
    fake_carla.OpendriveGenerationParameters = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "carla", fake_carla)

    agents_module = ModuleType("agents")
    navigation_module = ModuleType("agents.navigation")
    basic_agent_module = ModuleType("agents.navigation.basic_agent")
    behavior_agent_module = ModuleType("agents.navigation.behavior_agent")
    constant_velocity_module = ModuleType("agents.navigation.constant_velocity_agent")
    basic_agent_module.BasicAgent = _FakeAgent
    behavior_agent_module.BehaviorAgent = _FakeAgent
    constant_velocity_module.ConstantVelocityAgent = _FakeAgent
    monkeypatch.setitem(sys.modules, "agents", agents_module)
    monkeypatch.setitem(sys.modules, "agents.navigation", navigation_module)
    monkeypatch.setitem(sys.modules, "agents.navigation.basic_agent", basic_agent_module)
    monkeypatch.setitem(sys.modules, "agents.navigation.behavior_agent", behavior_agent_module)
    monkeypatch.setitem(
        sys.modules,
        "agents.navigation.constant_velocity_agent",
        constant_velocity_module,
    )

    sys.modules.pop("carla_agent_wrapper.carla_agent", None)
    return importlib.import_module("carla_agent_wrapper.carla_agent")


def test_clear_dynamic_actors_only_destroys_runtime_actor_types() -> None:
    from carla_agent_wrapper.lifecycle import clear_dynamic_actors

    vehicle = _FakeActor(actor_id=1, type_id="vehicle.tesla.model3")
    walker = _FakeActor(actor_id=2, type_id="walker.pedestrian.0001")
    controller = _FakeActor(actor_id=3, type_id="controller.ai.walker")
    sensor = _FakeActor(actor_id=4, type_id="sensor.other.collision")
    traffic_light = _FakeActor(actor_id=5, type_id="traffic.traffic_light")
    static_prop = _FakeActor(actor_id=6, type_id="static.prop.streetbarrier")
    world = _FakeWorld(
        [vehicle, walker, controller, sensor, traffic_light, static_prop],
        synchronous_mode=True,
        fixed_delta_seconds=0.05,
    )
    traffic_manager = _FakeTrafficManager()
    client = _FakeClient(world, traffic_manager=traffic_manager)

    clear_dynamic_actors(world, client=client, traffic_manager_port=8000)

    assert world.settings.synchronous_mode is False
    assert world.settings.fixed_delta_seconds is None
    assert world.applied_settings is world.settings
    assert traffic_manager.sync_calls == []
    assert vehicle.destroy_calls == 1
    assert walker.destroy_calls == 1
    assert controller.destroy_calls == 1
    assert sensor.destroy_calls == 1
    assert traffic_light.destroy_calls == 0
    assert static_prop.destroy_calls == 0


def test_clear_dynamic_actors_can_manage_traffic_manager_when_enabled() -> None:
    from carla_agent_wrapper.lifecycle import clear_dynamic_actors

    world = _FakeWorld([], synchronous_mode=True, fixed_delta_seconds=0.05)
    traffic_manager = _FakeTrafficManager()
    client = _FakeClient(world, traffic_manager=traffic_manager)

    clear_dynamic_actors(
        world,
        client=client,
        traffic_manager_port=8000,
        manage_traffic_manager=True,
    )

    assert traffic_manager.sync_calls == [False]


def test_destroy_actor_treats_false_return_as_failure() -> None:
    from carla_agent_wrapper.lifecycle import destroy_actor

    actor = _DestroyFalseActor(actor_id=1)

    assert destroy_actor(actor) is False
    assert actor.destroy_calls == 1


def test_reset_does_not_wrap_unclassified_runtime_error(carla_agent_module) -> None:
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)
    calls = []
    adapter._finalized = False
    adapter._finalize = lambda: calls.append("finalize")
    adapter._ensure_world = lambda map_name: (_ for _ in ()).throw(RuntimeError("world failed"))

    request = SimpleNamespace(
        output_dir="run",
        scenario_pack=SimpleNamespace(map_name="Town01"),
        initial_observation=ObservationData(ego=_state(carla_agent_module, 0.0)),
    )

    with pytest.raises(RuntimeError, match="world failed"):
        adapter.reset(request)

    assert calls == ["finalize"]


def test_reset_finalizes_partial_state_for_classified_av_error(carla_agent_module) -> None:
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)
    calls = []
    adapter._finalized = False
    adapter._finalize = lambda: calls.append("finalize")
    adapter._ensure_world = lambda map_name: (_ for _ in ()).throw(
        carla_agent_module.InvalidAvRequest("bad scenario")
    )

    request = SimpleNamespace(
        output_dir="run",
        scenario_pack=SimpleNamespace(map_name="Town01"),
        initial_observation=ObservationData(ego=_state(carla_agent_module, 0.0)),
    )

    with pytest.raises(carla_agent_module.InvalidAvRequest, match="bad scenario"):
        adapter.reset(request)

    assert calls == ["finalize", "finalize"]


def test_init_rejects_invalid_behavior(carla_agent_module) -> None:
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)
    adapter._finalized = True

    request = SimpleNamespace(output_dir="out", config={"behavior": "fast"}, dt=0.05)

    with pytest.raises(carla_agent_module.InvalidAvRequest, match="Unsupported CARLA behavior"):
        adapter.init(request)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"target_speed_is_mps": True}, "was removed"),
        ({"yaw_sign": 1.0}, "yaw_sign must be -1"),
    ],
)
def test_init_rejects_noncanonical_legacy_config(carla_agent_module, config, message) -> None:
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)
    adapter._finalized = True

    with pytest.raises(carla_agent_module.InvalidAvRequest, match=message):
        adapter.init(SimpleNamespace(output_dir="out", config=config, dt=0.05))


def test_init_requires_positive_finite_dt(carla_agent_module) -> None:
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)
    adapter._finalized = True

    with pytest.raises(carla_agent_module.InvalidAvRequest, match="positive"):
        adapter.init(SimpleNamespace(output_dir="out", config={}, dt=0.0))


def test_build_agent_configures_local_planner_completion_distance(carla_agent_module) -> None:
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)
    adapter._vehicle = object()
    adapter._map = object()
    adapter._agent_type = "behavior"
    adapter._behavior = "aggressive"
    adapter._BehaviorAgent = _FakeAgent
    adapter._local_planner_base_min_distance = 0.25
    adapter._local_planner_distance_ratio = 0.0
    adapter._route_sampling_resolution = 0.5
    adapter._follow_speed_limits = False
    adapter._ignore_traffic_lights = False
    adapter._ignore_stop_signs = False
    adapter._ignore_vehicles = False

    agent = adapter._build_agent(target_speed_kmh=50.0)

    assert agent._local_planner._base_min_distance == 0.25
    assert agent._local_planner._distance_ratio == 0.0
    assert agent._local_planner._min_distance == 0.25
    assert agent.kwargs["opt_dict"] == {
        "sampling_resolution": 0.5,
        "base_min_distance": 0.25,
        "distance_ratio": 0.0,
    }
    assert agent._vehicle_obstacle_detected([], 12.0) == ("native", [], 12.0)


def test_route_is_forced_to_end_at_destination_waypoint(carla_agent_module) -> None:
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)

    def waypoint(x):
        return SimpleNamespace(
            transform=SimpleNamespace(location=SimpleNamespace(x=x, y=0.0, z=0.0))
        )

    route_queue = [(waypoint(196.0), "LANEFOLLOW")]
    adapter._agent = SimpleNamespace(_local_planner=SimpleNamespace(_waypoints_queue=route_queue))
    end_wp = waypoint(200.0)

    adapter._ensure_route_ends_at_waypoint(end_wp)
    adapter._ensure_route_ends_at_waypoint(end_wp)

    assert route_queue == [
        (route_queue[0][0], "LANEFOLLOW"),
        (end_wp, "LANEFOLLOW"),
    ]


def test_step_returns_step_response(carla_agent_module) -> None:
    adapter, world = _make_tracking_adapter(carla_agent_module)
    adapter._agent = _FakeAgent()
    ego = _state(carla_agent_module, 0.0)

    response = adapter.step(
        SimpleNamespace(observation=carla_agent_module.ObservationData(ego=ego))
    )

    assert isinstance(response, carla_agent_module.StepResponse)
    assert response.ctrl_cmd.payload["throttle"] == pytest.approx(0.0)
    assert world.tick_calls == 1


def test_step_rejects_missing_ego_state(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    adapter._agent = _FakeAgent()

    with pytest.raises(carla_agent_module.InvalidAvRequest, match="ego state"):
        adapter.step(SimpleNamespace(observation=None))


def test_should_quit_returns_response(carla_agent_module) -> None:
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)
    adapter._quit_flag = True
    adapter._quit_msg = "CARLA agent reached the destination."

    response = adapter.should_quit()

    assert isinstance(response, carla_agent_module.ShouldQuitResponse)
    assert response.should_quit is True
    assert response.msg == "CARLA agent reached the destination."


def test_step_sets_destination_reached_quit_message(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    adapter._agent = _DoneAgent()
    ego = _state(carla_agent_module, 0.0)

    adapter.step(SimpleNamespace(observation=carla_agent_module.ObservationData(ego=ego)))
    response = adapter.should_quit()

    assert response.should_quit is True
    assert response.msg == "CARLA agent reached the destination."


def test_step_ignores_agent_done_until_ego_is_within_goal_distance(
    carla_agent_module,
) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    adapter._agent = _DoneAgent()
    adapter._goal_position_xy = (10.0, 0.0)
    adapter._agent_done_distance = 1.0
    adapter._quit_flag = False
    adapter._quit_msg = ""

    adapter.step(
        SimpleNamespace(
            observation=_observation(carla_agent_module, ego_x=7.0), timestamp_ns=0
        )
    )
    assert adapter.should_quit().should_quit is False

    adapter.step(
        SimpleNamespace(
            observation=ObservationData(
                ego=_state(carla_agent_module, 9.5, time_ns=1)
            ),
            timestamp_ns=1,
        )
    )
    assert adapter.should_quit().should_quit is True


def test_step_does_not_wrap_broken_private_state(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    adapter._vehicle = None
    adapter._agent = _FakeAgent()
    ego = _state(carla_agent_module, 0.0)

    with pytest.raises(RuntimeError, match="Ego vehicle not found"):
        adapter.step(SimpleNamespace(observation=carla_agent_module.ObservationData(ego=ego)))


def test_route_failure_is_concrete_precondition_failure(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    adapter._finalized = True
    adapter._ensure_world = lambda map_name: None
    adapter._clear_dynamic_actors = lambda: None
    adapter._apply_world_settings = lambda: None
    vehicle = adapter._vehicle
    adapter._spawn_ego = lambda init_obs, sps: vehicle
    vehicle.get_transform = lambda: SimpleNamespace(
        location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        rotation=SimpleNamespace(yaw=0.0),
    )
    adapter._map = SimpleNamespace(
        get_waypoint=lambda location, project_to_road: SimpleNamespace(
            transform=SimpleNamespace(location=location),
            road_id=1,
            section_id=1,
            lane_id=1,
            s=0.0,
        )
    )
    adapter._agent_type = "behavior"
    adapter._BehaviorAgent = _RouteFailingAgent
    adapter._behavior = "normal"
    adapter._route_sampling_resolution = 0.5
    adapter._local_planner_base_min_distance = 1.0
    adapter._local_planner_distance_ratio = 0.0
    adapter._follow_speed_limits = False
    adapter._ignore_traffic_lights = False
    adapter._ignore_stop_signs = False
    adapter._ignore_vehicles = False
    adapter._random_destination = False
    adapter._target_speed = 0.0
    adapter._target_speed_kmh = None
    adapter._spawn_z_offset = 0.0
    adapter._steer_sign = 1.0

    pos = SimpleNamespace(x=1.0, y=0.0, z=0.0)
    sps = SimpleNamespace(
        map_name="Town01",
        ego=SimpleNamespace(goal_config=SimpleNamespace(position=pos), target_speed=0.0),
    )
    ego = _state(carla_agent_module, 0.0)

    with pytest.raises(carla_agent_module.AvPreconditionFailed, match="failed to plan"):
        adapter.reset(
            SimpleNamespace(
                output_dir="run",
                scenario_pack=sps,
                initial_observation=carla_agent_module.ObservationData(ego=ego),
            )
        )


def test_ensure_world_reuses_cached_opendrive_world(carla_agent_module, tmp_path) -> None:
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)
    world = _FakeWorld()
    opendrive_path = (tmp_path / "Town01.xodr").resolve()
    adapter._reuse_generated_world = True
    adapter._world = world
    adapter._loaded_map_name = "Town01"
    adapter._loaded_opendrive_path = opendrive_path
    adapter._xodr_root = tmp_path
    adapter._server_version = "test"
    adapter._client = _FakeClient(world)

    adapter._ensure_world("Town01")

    assert adapter._world is world
    assert adapter._client.generated is False


def test_blueprint_lookup_falls_back_to_filter(carla_agent_module) -> None:
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)
    fallback_bp = _FakeBlueprint()
    bp_lib = _FakeBlueprintLibrary(filter_results={"vehicle.*bus*": [fallback_bp]})
    adapter._world = SimpleNamespace(get_blueprint_library=lambda: bp_lib)

    bp = adapter._pick_blueprint(carla_agent_module.RoadObjectType.BUS)

    assert bp is fallback_bp
    assert bp_lib.find_calls == ["vehicle.mitsubishi.fusorosa"]
    assert bp_lib.filter_calls == ["vehicle.*bus*"]


def _make_tracking_adapter(carla_agent_module):
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)
    bp = _FakeBlueprint()
    world = _FakeActorWorld(_FakeBlueprintLibrary(filter_results={"vehicle.*": [bp]}))
    adapter._world = world
    adapter._carla = SimpleNamespace(
        Location=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
        Rotation=lambda pitch, yaw, roll: SimpleNamespace(pitch=pitch, yaw=yaw, roll=roll),
        Transform=lambda location, rotation: SimpleNamespace(location=location, rotation=rotation),
        Vector3D=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
    )
    adapter._vehicle = _FakeMovingActor(actor_id=1)
    adapter._sync = True
    adapter._spawn_z_offset = 0.0
    adapter._coordinate_y_sign = 1.0
    adapter._yaw_sign = 1.0
    adapter._yaw_offset_deg = 0.0
    adapter._steer_sign = -1.0
    adapter._spawned_actor_ids = set()
    adapter._other_actors = []
    adapter._other_actor_types = []
    adapter._other_actors_by_key = {}
    adapter._other_actor_types_by_key = {}
    adapter._using_tracking_ids = False
    adapter._last_timestamp_ns = None
    adapter._ego_shape = None
    adapter._shapes_by_tracking_id = {}
    adapter._blueprint_dimensions = {}
    adapter._geometry_warnings = set()
    adapter._rng_seed = 0
    adapter._rng = random.Random(0)
    return adapter, world


def _kinematic(x):
    return SimpleNamespace(x=x, y=0.0, z=0.0, yaw=0.0, speed=0.0, yaw_rate=0.0)


_DEFAULT_SHAPE = object()


def _state(carla_agent_module, x, *, shape=_DEFAULT_SHAPE, time_ns=0):
    if shape is _DEFAULT_SHAPE:
        shape = _box()
    return ObjectStateData(
        type=carla_agent_module.RoadObjectType.CAR,
        kinematic=ObjectKinematicData(
            time_ns=time_ns,
            x=x,
            y=0.0,
            z=0.0,
            yaw=0.0,
            speed=0.0,
            yaw_rate=0.0,
        ),
        shape=shape,
    )


def _observation(carla_agent_module, agents=(), *, ego_x=0.0):
    return ObservationData(ego=_state(carla_agent_module, ego_x), agents=list(agents))


def _agent(
    carla_agent_module,
    x,
    *,
    tracking_id=None,
    entity_name=None,
    shape=_DEFAULT_SHAPE,
):
    return ObservedAgentData(
        state=_state(carla_agent_module, x, shape=shape),
        tracking_id=tracking_id,
        entity_name=entity_name,
    )


def test_tracking_ids_preserve_actor_identity_when_order_changes(carla_agent_module) -> None:
    adapter, world = _make_tracking_adapter(carla_agent_module)
    obj_a = _agent(carla_agent_module, 1.0, tracking_id=0)
    obj_b = _agent(carla_agent_module, 2.0, tracking_id=22)

    adapter._update_and_tick(_observation(carla_agent_module, [obj_a, obj_b]))
    first_actors = dict(adapter._other_actors_by_key)
    adapter._update_and_tick(_observation(carla_agent_module, [obj_b, obj_a]))

    assert len(world.spawned) == 2
    assert set(first_actors) == {0, 22}
    assert adapter._other_actors_by_key == first_actors


def test_missing_tracking_ids_recreate_actors_without_index_identity(carla_agent_module) -> None:
    adapter, world = _make_tracking_adapter(carla_agent_module)
    obj_a = _agent(carla_agent_module, 1.0)
    obj_b = _agent(carla_agent_module, 2.0)

    adapter._update_and_tick(_observation(carla_agent_module, [obj_a, obj_b]))
    first_actors = list(adapter._other_actors)
    adapter._update_and_tick(_observation(carla_agent_module, [obj_b, obj_a]))

    assert len(world.spawned) == 4
    assert adapter._other_actors_by_key == {}
    assert all(actor.destroy_calls == 1 for actor in first_actors)


def test_mixed_tracking_ids_use_stateless_behavior(carla_agent_module) -> None:
    adapter, world = _make_tracking_adapter(carla_agent_module)
    tracked = _agent(carla_agent_module, 1.0, tracking_id=10)
    untracked = _agent(carla_agent_module, 2.0)

    adapter._update_and_tick(_observation(carla_agent_module, [tracked, untracked]))
    adapter._update_and_tick(_observation(carla_agent_module, [untracked, tracked]))

    assert len(world.spawned) == 4
    assert adapter._other_actors_by_key == {}


def test_missing_id_frame_removes_previously_tracked_actors(carla_agent_module) -> None:
    adapter, world = _make_tracking_adapter(carla_agent_module)
    adapter._update_and_tick(
        _observation(carla_agent_module, [_agent(carla_agent_module, 1.0, tracking_id=10)])
    )
    tracked_actor = adapter._other_actors_by_key[10]

    adapter._update_and_tick(_observation(carla_agent_module, [_agent(carla_agent_module, 2.0)]))

    assert len(world.spawned) == 2
    assert tracked_actor.destroy_calls == 1
    assert adapter._other_actors_by_key == {}


def test_tracking_ids_create_new_and_remove_disappeared_actors(carla_agent_module) -> None:
    adapter, world = _make_tracking_adapter(carla_agent_module)
    first = _agent(carla_agent_module, 1.0, tracking_id=10)
    added = _agent(carla_agent_module, 2.0, tracking_id=20)

    adapter._update_and_tick(_observation(carla_agent_module, [first]))
    first_actor = adapter._other_actors_by_key[10]
    adapter._update_and_tick(_observation(carla_agent_module, [added]))

    assert len(world.spawned) == 2
    assert first_actor.destroy_calls == 1
    assert set(adapter._other_actors_by_key) == {20}


def test_tracking_id_type_change_replaces_actor(carla_agent_module) -> None:
    adapter, world = _make_tracking_adapter(carla_agent_module)
    original = _agent(carla_agent_module, 1.0, tracking_id=10)
    replacement = ObservedAgentData(
        state=ObjectStateData(
            type=carla_agent_module.RoadObjectType.PEDESTRIAN,
            kinematic=ObjectKinematicData(x=1.0),
        ),
        tracking_id=10,
    )
    world.blueprints.filter_results["walker.pedestrian.*"] = [_FakeBlueprint()]

    adapter._update_and_tick(_observation(carla_agent_module, [original]))
    original_actor = adapter._other_actors_by_key[10]
    adapter._update_and_tick(_observation(carla_agent_module, [replacement]))

    assert len(world.spawned) == 2
    assert original_actor.destroy_calls == 1
    assert adapter._other_actor_types_by_key[10] == carla_agent_module.RoadObjectType.PEDESTRIAN


def test_duplicate_tracking_ids_are_rejected(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    agents = [
        _agent(carla_agent_module, 1.0, tracking_id=7),
        _agent(carla_agent_module, 2.0, tracking_id=7),
    ]

    with pytest.raises(carla_agent_module.InvalidAvRequest, match="duplicate"):
        adapter._update_and_tick(_observation(carla_agent_module, agents))


def test_entity_names_are_optional_and_not_used_as_identity(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    named = _agent(carla_agent_module, 1.0, entity_name="NPC")
    unnamed = _agent(carla_agent_module, 2.0)

    adapter._update_and_tick(_observation(carla_agent_module, [named, unnamed]))

    assert len(adapter._other_actors) == 2
    assert adapter._other_actors_by_key == {}


def test_spawn_retries_above_observation_when_initial_spawn_collides(
    carla_agent_module,
) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    bp = _FakeBlueprint()
    world = _FailFirstSpawnWorld(_FakeBlueprintLibrary(filter_results={"vehicle.*": [bp]}))
    adapter._world = world
    adapter._spawn_z_offset = 0.0
    obj = _agent(carla_agent_module, 1.0)

    adapter._update_and_tick(_observation(carla_agent_module, [obj]))

    actor = adapter._other_actors[0]
    assert len(world.spawn_attempts) == 2
    assert world.spawn_attempts[0].location.z == 0.0
    assert world.spawn_attempts[1].location.z == 5.0
    assert actor.transforms[-1].location.z == 0.0


def test_only_non_ego_observation_actors_are_kinematic_before_tick(
    carla_agent_module,
) -> None:
    adapter, world = _make_tracking_adapter(carla_agent_module)
    ego_actor = adapter._vehicle
    obj = _agent(carla_agent_module, 1.0, tracking_id=1)

    adapter._update_and_tick(_observation(carla_agent_module, [obj]))

    other_actor = adapter._other_actors_by_key[1]
    assert ego_actor.simulate_physics_calls == []
    assert ego_actor.enable_gravity_calls == []
    assert other_actor.simulate_physics_calls[-1] is False
    assert other_actor.enable_gravity_calls[-1] is False
    assert world.tick_calls == 1


def test_agent_order_does_not_change_ego_pose(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    agents = [
        _agent(carla_agent_module, 1.0, tracking_id=1),
        _agent(carla_agent_module, 2.0, tracking_id=2),
    ]

    adapter._update_and_tick(_observation(carla_agent_module, agents, ego_x=50.0))
    adapter._update_and_tick(_observation(carla_agent_module, reversed(agents), ego_x=50.0))

    assert [transform.location.x for transform in adapter._vehicle.transforms] == [50.0, 50.0]


def test_bounding_box_center_and_actor_origin_offsets_are_composed(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    shape = ShapeData(
        type=ShapeType.BOUNDING_BOX,
        dimensions=ShapeDimensionData(x=4.0, y=2.0, z=1.5),
        center=ShapeCenterPoseData(x=2.0, y=1.0, z=0.5, yaw=1.5707963267948966),
        reference_point="rear_axle",
    )
    observed = _agent(carla_agent_module, 10.0, tracking_id=9, shape=shape)

    adapter._update_and_tick(_observation(carla_agent_module, [observed]))
    actor = adapter._other_actors_by_key[9]
    actor.bounding_box.location = SimpleNamespace(x=0.5, y=0.0, z=0.0)
    adapter._update_and_tick(_observation(carla_agent_module, [observed]))
    transform = actor.transforms[-1]

    assert transform.location.x == pytest.approx(12.0)
    assert transform.location.y == pytest.approx(0.5)
    assert transform.location.z == pytest.approx(0.5)
    assert transform.rotation.yaw == pytest.approx(90.0)


def test_reset_other_actor_state_clears_tracking(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    adapter._update_and_tick(
        _observation(carla_agent_module, [_agent(carla_agent_module, 1.0, tracking_id=4)])
    )

    adapter._reset_other_actor_state()

    assert adapter._other_actors == []
    assert adapter._other_actors_by_key == {}
    assert adapter._spawned_actor_ids == set()
    assert adapter._using_tracking_ids is False


def _box(*, length=4.0, width=2.0, height=1.5, center=None):
    return ShapeData(
        type=ShapeType.BOUNDING_BOX,
        dimensions=ShapeDimensionData(x=length, y=width, z=height),
        center=center or ShapeCenterPoseData(),
        reference_point="test_reference",
    )


def test_step_requires_matching_monotonic_simulation_timestamps(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    adapter._agent = _FakeAgent()
    first = ObservationData(ego=_state(carla_agent_module, 0.0, time_ns=0))
    second = ObservationData(ego=_state(carla_agent_module, 0.0, time_ns=10))

    adapter.step(SimpleNamespace(observation=first, timestamp_ns=0))
    adapter.step(SimpleNamespace(observation=second, timestamp_ns=10))

    with pytest.raises(carla_agent_module.InvalidAvRequest, match="increase strictly"):
        adapter.step(SimpleNamespace(observation=second, timestamp_ns=10))

    mismatched = ObservationData(ego=_state(carla_agent_module, 0.0, time_ns=11))
    with pytest.raises(carla_agent_module.InvalidAvRequest, match="must equal"):
        adapter.step(SimpleNamespace(observation=mismatched, timestamp_ns=12))


def test_observation_rejects_nonfinite_kinematic_and_invalid_shape(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    bad_state = _state(carla_agent_module, math.nan)
    with pytest.raises(carla_agent_module.InvalidAvRequest, match="finite"):
        adapter._prepare_observation(ObservationData(ego=bad_state))

    bad_shape = _box(length=0.0)
    with pytest.raises(carla_agent_module.InvalidAvRequest, match="positive"):
        adapter._prepare_observation(
            ObservationData(ego=_state(carla_agent_module, 0.0, shape=bad_shape))
        )


def test_tracked_shape_is_cached_and_shape_mutation_is_rejected(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    original = _agent(carla_agent_module, 1.0, tracking_id=8, shape=_box())
    adapter._prepare_observation(_observation(carla_agent_module, [original]))

    omitted = _agent(carla_agent_module, 2.0, tracking_id=8, shape=None)
    prepared = adapter._prepare_observation(_observation(carla_agent_module, [omitted]))
    assert prepared.agents[0].state.shape == original.state.shape

    changed = _agent(carla_agent_module, 2.0, tracking_id=8, shape=_box(length=5.0))
    with pytest.raises(carla_agent_module.InvalidAvRequest, match="shape changed"):
        adapter._prepare_observation(_observation(carla_agent_module, [changed]))


def test_first_observation_requires_shape(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    observation = ObservationData(ego=_state(carla_agent_module, 0.0, shape=None))

    with pytest.raises(carla_agent_module.InvalidAvRequest, match="shape is required"):
        adapter._prepare_observation(observation)


def test_unsupported_shape_is_rejected(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    shape = ShapeData(
        type=ShapeType.CYLINDER,
        dimensions=ShapeDimensionData(x=1.5, z=2.0),
        reference_point="test_reference",
    )

    with pytest.raises(carla_agent_module.InvalidAvRequest, match="BOUNDING_BOX"):
        adapter._prepare_observation(
            ObservationData(ego=_state(carla_agent_module, 0.0, shape=shape))
        )


def test_blueprint_matching_uses_nearest_dimensions_and_caches_measurements(
    carla_agent_module, caplog
) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    far = _FakeBlueprint("vehicle.far", (7.0, 3.0, 2.5))
    near = _FakeBlueprint("vehicle.near", (4.2, 2.1, 1.6))
    world = _FakeActorWorld(
        _FakeBlueprintLibrary(filter_results={"vehicle.*": [far, near]})
    )
    adapter._world = world
    state = _state(carla_agent_module, 0.0, shape=_box(length=4.0, width=2.0, height=1.5))

    with caplog.at_level("WARNING"):
        assert adapter._pick_blueprint_for_state(state) is near
        assert adapter._pick_blueprint_for_state(state) is near

    assert len(world.probes) == 2
    assert caplog.text.count("nearest CARLA geometry") == 1


def test_control_validation_and_brake_priority(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    adapter._agent = SimpleNamespace(
        run_step=lambda: SimpleNamespace(throttle=0.8, brake=0.5, steer=0.25)
    )
    response = adapter.step(
        SimpleNamespace(
            observation=ObservationData(ego=_state(carla_agent_module, 0.0)), timestamp_ns=0
        )
    )
    assert response.ctrl_cmd.payload == {"throttle": 0.0, "brake": 0.5, "steer": -0.25}

    adapter._last_timestamp_ns = None
    adapter._agent = SimpleNamespace(
        run_step=lambda: SimpleNamespace(throttle=1.1, brake=0.0, steer=0.0)
    )
    with pytest.raises(carla_agent_module.AvPreconditionFailed, match="throttle"):
        adapter.step(
            SimpleNamespace(
                observation=ObservationData(ego=_state(carla_agent_module, 0.0)),
                timestamp_ns=0,
            )
        )


def test_target_speed_uses_canonical_mps(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    adapter._target_speed = None
    adapter._target_speed_kmh = None
    sps = SimpleNamespace(ego=SimpleNamespace(target_speed=10.0))
    assert adapter._get_target_speed_kmh(sps) == pytest.approx(36.0)

    adapter._target_speed_kmh = 42.0
    assert adapter._get_target_speed_kmh(sps) == pytest.approx(42.0)


def test_destroy_spawned_actors_handles_none_and_clears_state(carla_agent_module) -> None:
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)
    vehicle = _FakeActor(actor_id=1)
    other = _FakeActor(actor_id=2)
    world = _FakeWorld([vehicle, other], synchronous_mode=True, fixed_delta_seconds=0.05)
    adapter._world = world
    adapter._client = _FakeClient(world)
    adapter._traffic_manager_port = 8000
    adapter._vehicle = vehicle
    adapter._other_actors = [None, other]
    adapter._other_actor_types = [object(), object()]
    adapter._spawned_actor_ids = {1}

    adapter._destroy_spawned_actors()

    assert vehicle.destroy_calls == 1
    assert other.destroy_calls == 1
    assert adapter._vehicle is None
    assert adapter._other_actors == []
    assert adapter._other_actor_types == []
    assert adapter._spawned_actor_ids == set()

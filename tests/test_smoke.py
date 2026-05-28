import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


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
    def __init__(self):
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
        self.tick_calls = 0

    def get_blueprint_library(self):
        return self.blueprints

    def try_spawn_actor(self, blueprint, transform):
        actor = _FakeMovingActor(actor_id=len(self.spawned) + 10)
        self.spawned.append((actor, blueprint, transform))
        self.actors.append(actor)
        return actor

    def tick(self):
        self.tick_calls += 1


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
    def __init__(self, actor_id, type_id="vehicle.test"):
        super().__init__(actor_id, type_id)
        self.transforms = []

    def set_transform(self, transform):
        self.transforms.append(transform)

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
    assert traffic_manager.sync_calls == [False]
    assert vehicle.destroy_calls == 1
    assert walker.destroy_calls == 1
    assert controller.destroy_calls == 1
    assert sensor.destroy_calls == 1
    assert traffic_light.destroy_calls == 0
    assert static_prop.destroy_calls == 0


def test_destroy_actor_treats_false_return_as_failure() -> None:
    from carla_agent_wrapper.lifecycle import destroy_actor

    actor = _DestroyFalseActor(actor_id=1)

    assert destroy_actor(actor) is False
    assert actor.destroy_calls == 1


def test_reset_finalizes_previous_run_and_partial_state_on_failure(carla_agent_module) -> None:
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)
    calls = []
    adapter._finalized = False
    adapter._finalize = lambda: calls.append("finalize")
    adapter._ensure_world = lambda map_name: (_ for _ in ()).throw(RuntimeError("world failed"))

    request = SimpleNamespace(
        output_dir="run",
        scenario_pack=SimpleNamespace(map_name="Town01"),
        initial_observation=[],
    )

    with pytest.raises(RuntimeError, match="world failed"):
        adapter.reset(request)

    assert calls == ["finalize", "finalize"]


def test_init_rejects_invalid_behavior(carla_agent_module) -> None:
    adapter = carla_agent_module.CarlaAgentAV.__new__(carla_agent_module.CarlaAgentAV)
    adapter._finalized = True

    request = SimpleNamespace(output_dir="out", config={"behavior": "fast"}, dt=0.05)

    with pytest.raises(ValueError, match="Unsupported CARLA behavior"):
        adapter.init(request)


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
    adapter._spawned_actor_ids = set()
    adapter._other_actors = []
    adapter._other_actor_types = []
    adapter._other_actors_by_key = {}
    adapter._other_actor_types_by_key = {}
    return adapter, world


def _kinematic(x):
    return SimpleNamespace(x=x, y=0.0, z=0.0, yaw=0.0, speed=0.0, yaw_rate=0.0)


def test_index_identity_mode_reuses_actors_when_order_is_stable(carla_agent_module) -> None:
    adapter, world = _make_tracking_adapter(carla_agent_module)
    adapter._object_identity_mode = "index"
    ego = SimpleNamespace(type=carla_agent_module.RoadObjectType.CAR, kinematic=_kinematic(0.0))
    obj_a = SimpleNamespace(type=carla_agent_module.RoadObjectType.CAR, kinematic=_kinematic(1.0))
    obj_b = SimpleNamespace(type=carla_agent_module.RoadObjectType.CAR, kinematic=_kinematic(2.0))

    adapter._update_and_tick([ego, obj_a, obj_b])
    first_actors = dict(adapter._other_actors_by_key)
    adapter._update_and_tick([ego, obj_a, obj_b])

    assert len(world.spawned) == 2
    assert adapter._other_actors_by_key == first_actors


def test_provided_identity_mode_uses_stable_identity_when_order_changes(carla_agent_module) -> None:
    adapter, world = _make_tracking_adapter(carla_agent_module)
    adapter._object_identity_mode = "provided"

    def kin(x):
        return _kinematic(x)

    ego = SimpleNamespace(type=carla_agent_module.RoadObjectType.CAR, kinematic=kin(0.0))
    obj_a = SimpleNamespace(id="a", type=carla_agent_module.RoadObjectType.CAR, kinematic=kin(1.0))
    obj_b = SimpleNamespace(id="b", type=carla_agent_module.RoadObjectType.CAR, kinematic=kin(2.0))

    adapter._update_and_tick([ego, obj_a, obj_b])
    first_actors = dict(adapter._other_actors_by_key)
    adapter._update_and_tick([ego, obj_b, obj_a])

    assert len(world.spawned) == 2
    assert adapter._other_actors_by_key == first_actors


def test_stateless_identity_mode_recreates_actors_each_step(carla_agent_module) -> None:
    adapter, world = _make_tracking_adapter(carla_agent_module)
    adapter._object_identity_mode = "stateless"
    ego = SimpleNamespace(type=carla_agent_module.RoadObjectType.CAR, kinematic=_kinematic(0.0))
    obj = SimpleNamespace(type=carla_agent_module.RoadObjectType.CAR, kinematic=_kinematic(1.0))

    adapter._update_and_tick([ego, obj])
    first_actor = adapter._other_actors_by_key[("frame", 0)]
    adapter._update_and_tick([ego, obj])

    assert len(world.spawned) == 2
    assert adapter._other_actors_by_key[("frame", 0)] is not first_actor


def test_provided_identity_mode_requires_object_id(carla_agent_module) -> None:
    adapter, _world = _make_tracking_adapter(carla_agent_module)
    adapter._object_identity_mode = "provided"
    ego = SimpleNamespace(type=carla_agent_module.RoadObjectType.CAR, kinematic=_kinematic(0.0))
    obj = SimpleNamespace(type=carla_agent_module.RoadObjectType.CAR, kinematic=_kinematic(1.0))

    with pytest.raises(RuntimeError, match="object_identity_mode='provided'"):
        adapter._update_and_tick([ego, obj])


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

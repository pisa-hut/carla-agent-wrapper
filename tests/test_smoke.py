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


class _FakeActor:
    def __init__(self, actor_id, type_id="vehicle.test"):
        self.id = actor_id
        self.type_id = type_id
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1


class _FakeAgent:
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

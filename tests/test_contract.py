import ast
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from google.protobuf.json_format import MessageToDict
from pisa_api import empty_pb2
from pisa_api.av import GenericAvService, InitRequest, InitResponse
from pisa_api.conversions import init_response_from_proto, init_response_to_proto

ROOT = Path(__file__).resolve().parents[1]


def test_server_entry_point_passes_stable_name_and_version() -> None:
    tree = ast.parse((ROOT / "carla_agent_wrapper/server.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "serve_av_system"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert ast.literal_eval(keywords["name"]) == "carla-agent-wrapper"
    assert isinstance(keywords["version"], ast.Call)
    assert getattr(keywords["version"].func, "id", None) == "wrapper_version"


def test_ping_returns_wrapper_identity_and_version() -> None:
    service = GenericAvService(object(), name="carla-agent-wrapper", version="0.2.1")
    context = SimpleNamespace(peer=lambda: "test")
    pong = service.Ping(empty_pb2.Empty(), context)
    assert pong.msg == "carla-agent-wrapper alive"
    assert pong.name == "carla-agent-wrapper"
    assert pong.version == "0.2.1"


def test_wrapper_version_prefers_distribution_metadata(monkeypatch) -> None:
    from carla_agent_wrapper import version

    monkeypatch.setattr(version.metadata, "version", lambda name: "9.8.7")
    assert version.wrapper_version() == "9.8.7"


def test_wrapper_version_falls_back_to_pyproject(monkeypatch) -> None:
    from carla_agent_wrapper import version

    def missing(_name):
        raise version.metadata.PackageNotFoundError

    monkeypatch.setattr(version.metadata, "version", missing)
    assert version.wrapper_version() == "0.2.1"


@pytest.fixture
def av_module(monkeypatch):
    fake_carla = ModuleType("carla")
    fake_carla.Client = lambda host, port: None
    monkeypatch.setitem(sys.modules, "carla", fake_carla)

    modules = {
        "agents": ModuleType("agents"),
        "agents.navigation": ModuleType("agents.navigation"),
        "agents.navigation.basic_agent": ModuleType("agents.navigation.basic_agent"),
        "agents.navigation.behavior_agent": ModuleType("agents.navigation.behavior_agent"),
        "agents.navigation.constant_velocity_agent": ModuleType(
            "agents.navigation.constant_velocity_agent"
        ),
    }
    modules["agents.navigation.basic_agent"].BasicAgent = object
    modules["agents.navigation.behavior_agent"].BehaviorAgent = object
    modules["agents.navigation.constant_velocity_agent"].ConstantVelocityAgent = object
    monkeypatch.setitem(sys.modules, "agents", modules["agents"])
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("carla_agent_wrapper.carla_agent", None)
    return importlib.import_module("carla_agent_wrapper.carla_agent")


def _initialized_adapter(av_module):
    adapter = av_module.CarlaAgentAV.__new__(av_module.CarlaAgentAV)
    adapter._finalized = True
    adapter._server_version = "test"
    adapter._prepare_reused_server_state = lambda: None
    return adapter


@pytest.mark.parametrize(
    ("configured", "component_name", "canonical_agent_type"),
    [
        ("behavior", "behavior-agent", "behavior"),
        ("constant-velocity", "constant-velocity-agent", "constant_velocity"),
    ],
)
def test_init_returns_component_and_struct_safe_effective_config(
    av_module, configured, component_name, canonical_agent_type
) -> None:
    adapter = _initialized_adapter(av_module)
    response = adapter.init(
        InitRequest(
            dt=0.05,
            output_dir="/tmp/output",
            config={
                "agent_type": configured.upper(),
                "behavior": "CAUTIOUS",
                "target_speed": 4,
                "random_seed": 7,
            },
        )
    )

    assert isinstance(response, InitResponse)
    assert response.name == component_name
    effective = response.metadata["effective_config"]
    assert effective["agent_type"] == canonical_agent_type
    assert effective["target_speed"] == 4.0
    assert effective["random_seed"] == 7
    assert (effective.get("behavior") == "cautious") is (canonical_agent_type == "behavior")
    assert "dt" not in effective
    assert "output_dir" not in effective
    assert set(response.metadata) == {"effective_config"}

    proto = init_response_to_proto(response)
    assert MessageToDict(proto.metadata)["effective_config"]["agent_type"] == canonical_agent_type
    assert init_response_from_proto(proto) == response


def test_init_failure_does_not_return_success_response(av_module) -> None:
    adapter = _initialized_adapter(av_module)
    with pytest.raises(av_module.InvalidAvRequest, match="Unsupported CARLA agent_type"):
        adapter.init(InitRequest(dt=0.05, output_dir="/tmp/output", config={"agent_type": "nope"}))

import logging
from typing import Any

logger = logging.getLogger(__name__)

RUNTIME_ACTOR_PREFIXES = (
    "vehicle.",
    "walker.",
    "controller.ai.walker",
    "sensor.",
)


def is_dynamic_actor(actor: Any) -> bool:
    type_id = getattr(actor, "type_id", "")
    return any(type_id.startswith(prefix) for prefix in RUNTIME_ACTOR_PREFIXES)


def force_async_world_for_cleanup(
    world: Any,
    *,
    client: Any = None,
    traffic_manager_port: int = 8000,
    log: logging.Logger = logger,
) -> None:
    if world is None:
        return

    try:
        settings = world.get_settings()
        changed = False
        if getattr(settings, "synchronous_mode", False):
            settings.synchronous_mode = False
            changed = True
        if getattr(settings, "fixed_delta_seconds", None) is not None:
            settings.fixed_delta_seconds = None
            changed = True
        if changed:
            world.apply_settings(settings)
            log.info("Forced CARLA world to async mode for cleanup")
    except Exception:
        log.exception("Failed to force CARLA world to async mode for cleanup")

    if client is None:
        return

    try:
        client.get_trafficmanager(traffic_manager_port).set_synchronous_mode(False)
    except Exception:
        log.exception("Failed to force TrafficManager to async mode for cleanup")


def clear_dynamic_actors(
    world: Any,
    *,
    client: Any = None,
    traffic_manager_port: int = 8000,
    log: logging.Logger = logger,
) -> int:
    if world is None:
        return 0

    force_async_world_for_cleanup(
        world,
        client=client,
        traffic_manager_port=traffic_manager_port,
        log=log,
    )

    try:
        actors = list(world.get_actors())
    except Exception:
        log.exception("Failed to list CARLA actors for reset cleanup")
        return 0

    destroyed_count = 0
    for actor in actors:
        if not is_dynamic_actor(actor):
            continue
        if destroy_actor(actor, log=log, label="dynamic actor"):
            destroyed_count += 1

    if destroyed_count:
        log.info("Destroyed %s dynamic CARLA actors before reset", destroyed_count)
    return destroyed_count


def destroy_actor(actor: Any, *, log: logging.Logger = logger, label: str = "actor") -> bool:
    if actor is None:
        return False

    actor_id = getattr(actor, "id", "<unknown>")
    try:
        result = actor.destroy()
        if result is False:
            log.warning("CARLA reported failure while destroying %s %s", label, actor_id)
            return False
        return True
    except Exception:
        log.exception("Failed to destroy %s %s", label, actor_id)
        return False

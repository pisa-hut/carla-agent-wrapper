# CARLA Agent Wrapper

This package exposes the CARLA navigation agents through the PISA AV service contract.

## Ping and initialization identity

The server's `Pong.name` is the stable wrapper artifact identity
`carla-agent-wrapper`. `Pong.version` is the installed package/build version (currently
`0.2.1`), obtained from the `carla-agent-wrapper` distribution metadata with the local
`pyproject.toml` project version as a source-checkout fallback.

`InitResponse.name` identifies the CARLA agent implementation selected and successfully
initialized by the request config: `behavior-agent`, `basic-agent`, or
`constant-velocity-agent`. It is intentionally distinct from the wrapper artifact identity.

`InitResponse.metadata.effective_config` records the normalized wrapper-specific settings
that actually affect the initialized CARLA agent. For example:

```json
{
  "effective_config": {
    "agent_type": "behavior",
    "behavior": "cautious",
    "ego_role_name": "hero",
    "ego_bp_id": "vehicle.tesla.model3",
    "sync": true,
    "no_rendering": true,
    "random_destination": false,
    "follow_speed_limits": false,
    "ignore_traffic_lights": false,
    "ignore_stop_signs": false,
    "ignore_vehicles": false,
    "local_planner_base_min_distance": 3.0,
    "local_planner_distance_ratio": 0.5,
    "agent_done_distance": 1.0,
    "route_sampling_resolution": 3.0,
    "coordinate_y_sign": -1.0,
    "yaw_sign": -1.0,
    "steer_sign": -1.0,
    "yaw_offset_deg": 0.0,
    "spawn_z_offset": 3.0,
    "reuse_generated_world": true,
    "manage_traffic_manager_sync": false,
    "random_seed": 0
  }
}
```

This metadata is written to the execution manifest. Do not put secrets, tokens, or
credentials in wrapper config. Shared execution data such as `dt`, map identity, output
directory, and scenario identity is not duplicated in this metadata.

The wrapper and runner must both use a compatible PISA API with the Ping/Init contract;
this package requires `pisa-api>=0.4.1`.

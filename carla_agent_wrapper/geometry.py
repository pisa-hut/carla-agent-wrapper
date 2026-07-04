import math
from typing import Any

SHAPE_TOLERANCE = 1e-9


def finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def validate_state(state: Any, *, require_shape: bool, shape_types: Any) -> None:
    kin = state.kinematic
    for field in (
        "x",
        "y",
        "z",
        "yaw",
        "speed",
        "acceleration",
        "yaw_rate",
        "yaw_acceleration",
    ):
        finite(getattr(kin, field), f"kinematic.{field}")

    shape = state.shape
    if shape is None:
        if require_shape:
            raise ValueError("shape is required on an actor's first observation")
        return

    if not str(shape.reference_point).strip():
        raise ValueError("shape.reference_point must identify the kinematic origin")

    for field in ("x", "y", "z", "roll", "pitch", "yaw"):
        finite(getattr(shape.center, field), f"shape.center.{field}")

    if shape.type == shape_types.BOUNDING_BOX:
        for field in ("x", "y", "z"):
            value = finite(getattr(shape.dimensions, field), f"shape.dimensions.{field}")
            if value <= 0:
                raise ValueError(f"shape.dimensions.{field} must be positive")
    else:
        raise ValueError(
            f"unsupported shape type: {shape.type!r}; CARLA agents support BOUNDING_BOX"
        )


def normalized_vertices(vertices: Any) -> tuple[tuple[float, float, float], ...]:
    result = tuple(
        (
            finite(vertex.x, "shape.vertices.x"),
            finite(vertex.y, "shape.vertices.y"),
            finite(vertex.z, "shape.vertices.z"),
        )
        for vertex in vertices
    )
    if len(result) > 1 and result[0] == result[-1]:
        return result[:-1]
    return result


def shapes_equivalent(left: Any, right: Any) -> bool:
    if left.type != right.type or left.reference_point != right.reference_point:
        return False
    left_values = _shape_values(left)
    right_values = _shape_values(right)
    return len(left_values) == len(right_values) and all(
        math.isclose(a, b, rel_tol=SHAPE_TOLERANCE, abs_tol=SHAPE_TOLERANCE)
        for a, b in zip(left_values, right_values, strict=True)
    )


def _shape_values(shape: Any) -> tuple[float, ...]:
    center = shape.center
    dimensions = shape.dimensions
    values = (
        float(center.x),
        float(center.y),
        float(center.z),
        float(center.roll),
        float(center.pitch),
        float(center.yaw),
        float(dimensions.x),
        float(dimensions.y),
        float(dimensions.z),
    )
    return values + tuple(
        value for vertex in normalized_vertices(shape.vertices) for value in vertex
    )

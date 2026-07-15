"""Serialization helpers for RPC — round-trip Pydantic result types."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from llm_inspector.models.measurement import Measurement, MeasurementStatus

T = TypeVar("T", bound=BaseModel)


def model_to_dict(obj: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model to a JSON-safe dict."""
    return obj.model_dump(mode="json")


def model_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Deserialize a dict into a Pydantic model."""
    return cls.model_validate(data)


def measurement_to_dict(m: Measurement[Any]) -> dict[str, Any]:
    return {
        "value": m.value,
        "status": m.status.value,
        "source": m.source,
        "reason": m.reason,
    }


def measurement_from_dict(data: dict[str, Any]) -> Measurement[Any]:
    return Measurement(
        value=data.get("value"),
        status=MeasurementStatus(data.get("status", MeasurementStatus.UNAVAILABLE)),
        source=data.get("source"),
        reason=data.get("reason"),
    )

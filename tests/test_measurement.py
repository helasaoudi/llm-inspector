"""Tests for the Measurement[T] type — the core trust primitive."""

import pytest
from llm_inspector.models.measurement import Measurement, MeasurementStatus


def test_available_sets_correct_fields():
    m = Measurement[int].available(1_000_000, source="NVML")
    assert m.is_available is True
    assert m.value == 1_000_000
    assert m.source == "NVML"
    assert m.reason is None
    assert m.status == MeasurementStatus.AVAILABLE


def test_unavailable_sets_correct_fields():
    m = Measurement[int].unavailable("Runtime does not expose this.")
    assert m.is_available is False
    assert m.value is None
    assert m.reason == "Runtime does not expose this."
    assert m.source is None
    assert m.status == MeasurementStatus.UNAVAILABLE


def test_measurement_is_immutable():
    m = Measurement[int].available(42, source="test")
    with pytest.raises(Exception):
        m.value = 99  # type: ignore[misc]


def test_available_with_string_type():
    m = Measurement[str].available("FP16", source="cmdline --dtype")
    assert m.value == "FP16"
    assert m.is_available is True


def test_unavailable_with_string_type():
    m = Measurement[str].unavailable("Not exposed.")
    assert m.value is None
    assert m.is_available is False

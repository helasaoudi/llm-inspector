"""Tests for RPC wire protocol."""

from llm_inspector.rpc import SCHEMA_VERSION, CollectorRequest, CollectorResponse


class TestCollectorRequest:
    def test_defaults(self):
        req = CollectorRequest(collector="memory")
        assert req.schema_version == SCHEMA_VERSION
        assert req.collector == "memory"
        assert req.options == {}

    def test_round_trip(self):
        req = CollectorRequest(collector="model", options={"trace_layers": True})
        data = req.model_dump()
        restored = CollectorRequest.model_validate(data)
        assert restored.collector == "model"
        assert restored.options["trace_layers"] is True


class TestCollectorResponse:
    def test_success(self):
        resp = CollectorResponse(collector="memory", data={"gpu_allocated": {}})
        assert resp.succeeded

    def test_failure(self):
        resp = CollectorResponse(collector="memory", error="boom")
        assert not resp.succeeded

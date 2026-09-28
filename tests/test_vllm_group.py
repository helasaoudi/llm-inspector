"""Tests for vLLM TP worker group discovery."""

from __future__ import annotations

from unittest.mock import patch

from llm_inspector.utils.vllm_group import (
    discover_vllm_tp_group,
    is_vllm_tp_member,
    tp_rank,
    worker_label,
)


class TestVllmTpGroup:
    def test_non_member_returns_alone(self) -> None:
        with (
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.read_cmdline",
                return_value=["python", "app.py"],
            ),
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.read_name",
                return_value="python",
            ),
        ):
            assert discover_vllm_tp_group(10) == [10]
            assert is_vllm_tp_member(10) is False

    def test_discovers_sibling_workers_same_ppid(self) -> None:
        # PID 100 = Worker_TP0, 101 = Worker_TP1, both ppid=50
        def cmdline(pid: int) -> list[str]:
            return {
                100: ["VLLM::Worker_TP0"],
                101: ["VLLM::Worker_TP1"],
                50: ["vllm", "serve"],
            }.get(pid, [])

        def name(pid: int) -> str | None:
            return {
                100: "VLLM::Worker_TP0",
                101: "VLLM::Worker_TP1",
                50: "vllm",
            }.get(pid)

        def ppid(pid: int) -> int | None:
            return {100: 50, 101: 50, 50: 1}.get(pid)

        with (
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.read_cmdline",
                side_effect=cmdline,
            ),
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.read_name",
                side_effect=name,
            ),
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.read_ppid",
                side_effect=ppid,
            ),
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.list_all_pids",
                return_value=[50, 100, 101, 999],
            ),
        ):
            assert discover_vllm_tp_group(100) == [100, 101]
            assert discover_vllm_tp_group(101) == [100, 101]
            assert tp_rank(100) == 0
            assert tp_rank(101) == 1
            assert worker_label(100) == "Worker_TP0"

    def test_includes_engine_core_sibling(self) -> None:
        def cmdline(pid: int) -> list[str]:
            return {
                10: ["VLLM::EngineCore"],
                11: ["VLLM::Worker_TP0"],
                12: ["VLLM::Worker_TP1"],
            }.get(pid, [])

        def name(pid: int) -> str | None:
            return cmdline(pid)[0] if cmdline(pid) else None

        def ppid(pid: int) -> int | None:
            return 1  # all siblings

        with (
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.read_cmdline",
                side_effect=cmdline,
            ),
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.read_name",
                side_effect=name,
            ),
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.read_ppid",
                side_effect=ppid,
            ),
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.list_all_pids",
                return_value=[10, 11, 12],
            ),
        ):
            group = discover_vllm_tp_group(11)
            assert group == [10, 11, 12]  # EngineCore first, then TP0, TP1


class TestCollapseTpForPs:
    def test_collapses_engine_and_workers(self) -> None:
        from llm_inspector.models.enums import BackendKind, RuntimeKind
        from llm_inspector.models.measurement import Measurement
        from llm_inspector.models.report import InspectionReport
        from llm_inspector.models.results import HardwareResult, ModelResult, ProcessResult
        from llm_inspector.utils.vllm_group import collapse_tp_reports_for_ps

        def make(pid: int, name: str, vram: int | None, gpus: list[int]) -> InspectionReport:
            from llm_inspector.models.results import GpuAttachment

            devices = [GpuAttachment(index=g, process_vram_bytes=vram) for g in gpus]
            return InspectionReport(
                pid=pid,
                process=ProcessResult(
                    pid=pid,
                    cmdline=[name],
                    runtime_kind=RuntimeKind.VLLM,
                ),
                hardware=HardwareResult(
                    backend=BackendKind.CUDA if gpus else BackendKind.CPU,
                    device_index=gpus[0] if gpus else None,
                    vram_used_bytes=vram,
                    devices=devices,
                ),
                model=ModelResult(
                    name=Measurement.available("AceMath", "vllm.engine.model_config.model")
                )
                if "Worker" in name
                else None,
            )

        reports = [
            make(10, "VLLM::EngineCore", None, []),
            make(11, "VLLM::Worker_TP0", 7 * 1024**3, [0]),
            make(12, "VLLM::Worker_TP1", 7 * 1024**3, [1]),
        ]

        with (
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.read_cmdline",
                side_effect=lambda pid: {
                    10: ["VLLM::EngineCore"],
                    11: ["VLLM::Worker_TP0"],
                    12: ["VLLM::Worker_TP1"],
                }.get(pid, []),
            ),
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.read_name",
                side_effect=lambda pid: {
                    10: "VLLM::EngineCore",
                    11: "VLLM::Worker_TP0",
                    12: "VLLM::Worker_TP1",
                }.get(pid),
            ),
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.read_ppid",
                return_value=1,
            ),
            patch(
                "llm_inspector.utils.vllm_group.proc_utils.list_all_pids",
                return_value=[10, 11, 12],
            ),
        ):
            groups = collapse_tp_reports_for_ps(reports)

        assert len(groups) == 1
        rep, members = groups[0]
        assert len(members) == 3
        assert rep.pid in (11, 12)  # worker with model, not EngineCore

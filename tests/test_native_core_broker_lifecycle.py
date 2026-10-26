from __future__ import annotations

import os
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from wechat_decrypt_tool import native_core_broker


@pytest.fixture(autouse=True)
def isolated_diagnostic_repeats(monkeypatch):
    monkeypatch.setattr(native_core_broker, "_diagnostic_repeats", native_core_broker.OrderedDict())


def test_diagnostic_state_keeps_exit_signal_before_restart(monkeypatch):
    process = SimpleNamespace(pid=123, poll=lambda: -9)
    monkeypatch.setattr(native_core_broker, "_process", process)
    monkeypatch.setattr(native_core_broker, "_diagnostic_started", 100.0)
    monkeypatch.setattr(native_core_broker, "_diagnostic_queries", 456)
    monkeypatch.setattr(native_core_broker.time, "monotonic", lambda: 7800.0)
    state = native_core_broker.native_core_diagnostic_state()
    assert state["broker_pid"] == 123
    assert state["broker_exit_code"] == -9
    assert state["broker_signal"] == 9
    assert state["broker_uptime_seconds"] == 7700.0
    assert state["broker_query_count"] == 456
    assert not state["broker_alive"]


def test_diagnostic_log_omits_sql_errors_and_endpoint(monkeypatch, tmp_path):
    (tmp_path / "logs").mkdir()
    with monkeypatch.context() as scoped:
        scoped.setenv("WECHAT_TOOL_DATA_DIR", str(tmp_path))
        scoped.setattr(native_core_broker, "_process", SimpleNamespace(pid=123, poll=lambda: None))
        scoped.setattr(native_core_broker, "_owned_endpoint", "SECRET_ENDPOINT")
        native_core_broker.record_native_core_diagnostic(
            "native_query_error", status=-2, stage="fetch_query", sql="SECRET_SQL", error="SECRET_ERROR",
        )
    raw = (tmp_path / "logs/native-core-diagnostics.log").read_text()
    event = json.loads(raw)
    assert event["broker_alive"]
    assert event["stage"] == "fetch_query"
    assert event["status"] == -2
    assert "SECRET" not in raw
    assert "timestamp" in event


def test_unwritable_diagnostic_log_does_not_break_recovery(monkeypatch, tmp_path):
    (tmp_path / "logs").write_text("not a directory")
    monkeypatch.setenv("WECHAT_TOOL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(native_core_broker, "_process", None)
    native_core_broker.record_native_core_diagnostic("native_broker_started")


def test_repeated_query_errors_are_summarized_and_new_errors_are_immediate(monkeypatch, tmp_path):
    monkeypatch.setenv("WECHAT_TOOL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(native_core_broker, "_process", None)
    clock = [0.0]
    monkeypatch.setattr(native_core_broker.time, "monotonic", lambda: clock[0])
    def record(status=-15):
        native_core_broker.record_native_core_diagnostic(
            "native_query_generation_failed", stage="open_query", status=status, error_type="NativeCoreError",
        )
    record()
    for _ in range(100):
        record()
    record(-4)
    clock[0] = 300.0
    record()
    rows = [json.loads(line) for line in (tmp_path / "logs/native-core-diagnostics.log").read_text().splitlines()]
    assert [row["status"] for row in rows] == [-15, -4, -15]
    assert rows[-1]["suppressed_count"] == 100
    assert len(native_core_broker._diagnostic_repeats) == 2


def test_diagnostic_rotation_keeps_bounded_backups_and_does_not_touch_broker_stderr(monkeypatch, tmp_path):
    monkeypatch.setenv("WECHAT_TOOL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(native_core_broker, "_process", None)
    monkeypatch.setattr(native_core_broker, "_DIAGNOSTIC_LOG_MAX_BYTES", 1)
    logs = tmp_path / "logs"
    logs.mkdir()
    stderr = logs / "native-core-broker.log"
    stderr.write_text("broker stderr remains open\n")
    for index in range(8):
        native_core_broker.record_native_core_diagnostic("native_broker_started", attempt=index)
    assert stderr.read_text() == "broker stderr remains open\n"
    paths = sorted(logs.glob("native-core-diagnostics.log*"))
    assert len(paths) == 4
    assert [json.loads(path.read_text())["attempt"] for path in paths] == [7, 6, 5, 4]
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)


def test_diagnostic_repeat_cache_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setenv("WECHAT_TOOL_DATA_DIR", str(tmp_path))
    for status in range(150):
        native_core_broker.record_native_core_diagnostic("native_query_generation_failed", status=status)
    assert len(native_core_broker._diagnostic_repeats) == 128


def test_windows_and_macos_broker_first_start_has_a_sixty_second_default_budget() -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(native_core_broker.sys, "platform", "darwin"),
    ):
        assert native_core_broker._startup_timeout_seconds() == 60.0

    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(native_core_broker.sys, "platform", "win32"),
    ):
        assert native_core_broker._startup_timeout_seconds() == 60.0


def test_broker_startup_timeout_environment_still_overrides_platform_default() -> None:
    with (
        patch.dict(
            os.environ,
            {native_core_broker.ENV_NATIVE_CORE_STARTUP_TIMEOUT_MS: "7000"},
            clear=True,
        ),
        patch.object(native_core_broker.sys, "platform", "darwin"),
    ):
        assert native_core_broker._startup_timeout_seconds() == 7.0


def test_broker_log_tail_is_bounded_to_the_current_run(tmp_path: Path) -> None:
    log_path = tmp_path / "native-core-broker.log"
    previous = b"old sensitive output\n"
    log_path.write_bytes(
        previous
        + b"wechatdb_broker ready endpoint=\\\\.\\pipe\\secret\n"
        + b"Broker stopped: I/O error\n"
    )

    tail = native_core_broker._broker_log_tail(log_path, len(previous))

    assert "old sensitive output" not in tail
    assert "endpoint=<redacted>" in tail
    assert "Broker stopped: I/O error" in tail


def test_broker_startup_timeout_preserves_last_native_error() -> None:
    process = SimpleNamespace(poll=lambda: None)
    with (
        patch.object(
            native_core_broker.time,
            "monotonic",
            side_effect=[0.0, 0.0, 0.0, 1.0],
        ),
        patch.object(native_core_broker.time, "sleep"),
        patch.object(
            native_core_broker,
            "NativeCoreClient",
            side_effect=native_core_broker.NativeCoreUnavailableError("I/O error (-4)"),
        ),
    ):
        try:
            native_core_broker._wait_until_ready(
                process,
                client_path=Path("wechatdb_client.dll"),
                endpoint="test-endpoint",
                timeout=0.5,
            )
        except native_core_broker.NativeCoreUnavailableError as exc:
            assert "Last native error: I/O error (-4)" in str(exc)
        else:
            raise AssertionError("broker startup timeout must fail")


def test_broker_validates_controlled_distribution_against_client_artifact(
    tmp_path: Path,
) -> None:
    broker_path = tmp_path / "wechatdb_broker.exe"
    client_path = tmp_path / native_core_broker._client_name()
    broker_path.write_bytes(b"broker")
    client_path.write_bytes(b"client")
    manifest = object()
    process = SimpleNamespace(poll=lambda: None)
    original = (
        native_core_broker._process,
        native_core_broker._owned_endpoint,
        native_core_broker._owned_database_roots,
        native_core_broker._owned_database_disabled,
        native_core_broker._active_operations,
    )
    try:
        native_core_broker._process = None
        native_core_broker._owned_endpoint = ""
        native_core_broker._owned_database_roots = ()
        native_core_broker._owned_database_disabled = False
        native_core_broker._active_operations = 0
        with (
            patch.object(
                native_core_broker,
                "resolve_native_core_broker",
                return_value=broker_path,
            ),
            patch.object(
                native_core_broker,
                "_required_native_core_build_manifest",
                return_value=manifest,
            ) as require_manifest,
            patch.object(native_core_broker, "_new_endpoint", return_value="test-endpoint"),
            patch.object(native_core_broker, "_resolve_broker_trust_key", return_value=""),
            patch.object(native_core_broker, "_broker_child_environment", return_value={}),
            patch.object(
                native_core_broker,
                "_broker_log_file",
                return_value=native_core_broker.subprocess.DEVNULL,
            ),
            patch.object(native_core_broker.subprocess, "Popen", return_value=process),
            patch.object(native_core_broker, "_wait_until_ready"),
            patch.object(native_core_broker, "close_native_core_client"),
            patch.object(native_core_broker, "_install_owned_environment"),
        ):
            assert native_core_broker.ensure_native_core_broker(export_only=True) == (
                "test-endpoint"
            )

        require_manifest.assert_called_once_with(client_path)
    finally:
        (
            native_core_broker._process,
            native_core_broker._owned_endpoint,
            native_core_broker._owned_database_roots,
            native_core_broker._owned_database_disabled,
            native_core_broker._active_operations,
        ) = original


def test_last_export_only_operation_stops_owned_broker() -> None:
    original = (
        native_core_broker._process,
        native_core_broker._owned_database_disabled,
        native_core_broker._active_operations,
    )
    try:
        native_core_broker._process = object()
        native_core_broker._owned_database_disabled = True
        native_core_broker._active_operations = 1
        operation = native_core_broker.NativeCoreManagedOperation("test-endpoint")

        with patch.object(native_core_broker, "stop_native_core_broker") as stop:
            operation.close()

        stop.assert_called_once_with()
        assert native_core_broker._active_operations == 0
    finally:
        (
            native_core_broker._process,
            native_core_broker._owned_database_disabled,
            native_core_broker._active_operations,
        ) = original


@pytest.mark.parametrize("state", ["absent", "dead", "export", "database"])
def test_export_preference_selects_mode_without_replacing_live_broker(monkeypatch, state):
    process = None if state == "absent" else Mock()
    if process is not None:
        process.poll.return_value = 1 if state == "dead" else None
    monkeypatch.setattr(native_core_broker, "_process", process)
    monkeypatch.setattr(native_core_broker, "_owned_database_disabled", state == "export")
    monkeypatch.setattr(native_core_broker, "_active_operations", 0)
    with (
        patch.object(native_core_broker, "ensure_native_core_broker", return_value="endpoint") as ensure,
        patch.object(native_core_broker, "stop_native_core_broker"),
    ):
        with native_core_broker.managed_native_core_operation(prefer_export_only=True):
            assert native_core_broker._active_operations == 1
        assert native_core_broker._active_operations == 0
    ensure.assert_called_once_with(database_root=None, export_only=state != "database")


def test_strict_export_only_still_rejects_busy_database_broker(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(native_core_broker, "_process", process)
    monkeypatch.setattr(native_core_broker, "_owned_database_disabled", False)
    monkeypatch.setattr(native_core_broker, "_active_operations", 1)
    with pytest.raises(native_core_broker.NativeCoreUnavailableError, match="BUSY with 1"):
        native_core_broker.managed_native_core_operation(export_only=True)
    assert native_core_broker._active_operations == 1


@pytest.mark.parametrize("options", [{"export_only": True}, {"database_root": Path("unused")}])
def test_export_preference_rejects_conflicting_requirements(options):
    with pytest.raises(native_core_broker.NativeCoreProtocolError, match="cannot be combined"):
        native_core_broker.managed_native_core_operation(prefer_export_only=True, **options)

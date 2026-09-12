import ctypes
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── start_heartbeat_sentinel ──────────────────────────────────────────────────


def test_start_heartbeat_sentinel_noop_on_non_windows(monkeypatch):
    import litserve._win_shutdown_fix as mod

    monkeypatch.setattr(mod.sys, "platform", "linux")
    with patch.object(mod, "_create_process_no_window") as mock_spawn:
        mod.start_heartbeat_sentinel(1234, "/tmp/hb.tmp", 3.0)
    mock_spawn.assert_not_called()


def test_start_heartbeat_sentinel_writes_ps1_and_spawns(monkeypatch, tmp_path):
    import litserve._win_shutdown_fix as mod

    monkeypatch.setattr(mod.sys, "platform", "win32")

    with (
        patch.object(mod, "_create_process_no_window", return_value=True) as mock_spawn,
        patch("tempfile.gettempdir", return_value=str(tmp_path)),
    ):
        mod.start_heartbeat_sentinel(99999, r"C:\hb.tmp", 3.0)

    ps1 = tmp_path / "litserve_spawn_sentinel_99999.ps1"
    assert ps1.exists()
    content = ps1.read_text()
    assert "Invoke-WmiMethod" in content
    assert "Win32_Process" in content
    assert "99999" in content
    assert "3.0" in content
    assert "_child.py" in content

    mock_spawn.assert_called_once()
    spawn_arg = mock_spawn.call_args[0][0]
    assert spawn_arg.startswith("powershell.exe -NoProfile")
    assert str(ps1) in spawn_arg


def test_start_heartbeat_sentinel_escapes_single_quotes(monkeypatch, tmp_path):
    import litserve._win_shutdown_fix as mod

    monkeypatch.setattr(mod.sys, "platform", "win32")

    with (
        patch.object(mod, "_create_process_no_window", return_value=True),
        patch("tempfile.gettempdir", return_value=str(tmp_path)),
    ):
        mod.start_heartbeat_sentinel(12345, r"C:\o'malley\hb.tmp", 3.0)

    content = (tmp_path / "litserve_spawn_sentinel_12345.ps1").read_text()
    assert "''" in content  # single quotes doubled for PS single-quoted string


def test_start_heartbeat_sentinel_swallows_spawn_errors(monkeypatch, tmp_path):
    import litserve._win_shutdown_fix as mod

    monkeypatch.setattr(mod.sys, "platform", "win32")

    with (
        patch.object(mod, "_create_process_no_window", side_effect=RuntimeError("kaboom")),
        patch("tempfile.gettempdir", return_value=str(tmp_path)),
    ):
        mod.start_heartbeat_sentinel(99999, r"C:\hb.tmp", 3.0)  # must not raise


# ── _create_process_no_window ─────────────────────────────────────────────────


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only: tests ctypes.windll directly")
def test_create_process_no_window_returns_true_on_success(monkeypatch):
    k32 = MagicMock()
    k32.CreateProcessW.return_value = 1
    monkeypatch.setattr(ctypes, "windll", MagicMock(kernel32=k32), raising=False)

    from litserve._win_shutdown_fix import _create_process_no_window

    assert _create_process_no_window("cmd.exe") is True
    assert k32.CloseHandle.call_count == 2


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only: tests ctypes.windll directly")
def test_create_process_no_window_returns_false_on_failure(monkeypatch):
    k32 = MagicMock()
    k32.CreateProcessW.return_value = 0
    monkeypatch.setattr(ctypes, "windll", MagicMock(kernel32=k32), raising=False)

    from litserve._win_shutdown_fix import _create_process_no_window

    assert _create_process_no_window("cmd.exe") is False
    k32.CloseHandle.assert_not_called()


# ── _child._alive ─────────────────────────────────────────────────────────────


def test_alive_true_when_handle_returned(monkeypatch):
    k32 = MagicMock()
    k32.OpenProcess.return_value = 0x1234
    monkeypatch.setattr(ctypes, "windll", MagicMock(kernel32=k32), raising=False)

    from litserve._win_shutdown_fix import _child

    assert _child._alive(123) is True
    k32.CloseHandle.assert_called_once_with(0x1234)


def test_alive_false_when_no_handle(monkeypatch):
    k32 = MagicMock()
    k32.OpenProcess.return_value = 0
    monkeypatch.setattr(ctypes, "windll", MagicMock(kernel32=k32), raising=False)

    from litserve._win_shutdown_fix import _child

    assert _child._alive(123) is False
    k32.CloseHandle.assert_not_called()


# ── _child._kill_subtree ──────────────────────────────────────────────────────


def test_kill_subtree_invalid_snapshot_is_noop(monkeypatch):
    k32 = MagicMock()
    k32.CreateToolhelp32Snapshot.return_value = ctypes.c_void_p(-1).value
    monkeypatch.setattr(ctypes, "windll", MagicMock(kernel32=k32), raising=False)

    from litserve._win_shutdown_fix import _child

    _child._kill_subtree(100)

    k32.Process32First.assert_not_called()
    k32.OpenProcess.assert_not_called()
    k32.TerminateProcess.assert_not_called()


def test_kill_subtree_empty_snapshot_kills_only_root(monkeypatch):
    k32 = MagicMock()
    k32.CreateToolhelp32Snapshot.return_value = 42
    k32.Process32First.return_value = 0  # empty snapshot: no entries
    k32.OpenProcess.return_value = 0xAB
    monkeypatch.setattr(ctypes, "windll", MagicMock(kernel32=k32), raising=False)

    from litserve._win_shutdown_fix import _child

    _child._kill_subtree(100)

    k32.CloseHandle.assert_any_call(42)  # snapshot handle closed
    k32.OpenProcess.assert_called_once()
    assert k32.OpenProcess.call_args[0][2] == 100  # root pid targeted
    k32.TerminateProcess.assert_called_once()


# ── _child.main ───────────────────────────────────────────────────────────────


def test_main_kills_when_pid_dies(monkeypatch):
    from litserve._win_shutdown_fix import _child

    mock_kill = MagicMock()
    monkeypatch.setattr(_child, "_alive", MagicMock(return_value=False))
    monkeypatch.setattr(_child, "_kill_subtree", mock_kill)
    monkeypatch.setattr(_child.time, "sleep", MagicMock())
    monkeypatch.setattr(sys, "argv", ["_child.py", "9999", "/tmp/hb.tmp", "3.0"])

    _child.main()

    mock_kill.assert_called_once_with(9999)


def test_main_kills_when_heartbeat_file_missing(monkeypatch):
    from litserve._win_shutdown_fix import _child

    mock_kill = MagicMock()
    monkeypatch.setattr(_child, "_alive", MagicMock(return_value=True))
    monkeypatch.setattr(_child, "_kill_subtree", mock_kill)
    monkeypatch.setattr(_child.time, "sleep", MagicMock())
    monkeypatch.setattr(_child.time, "time", MagicMock(return_value=1000.0))
    monkeypatch.setattr(_child.os.path, "getmtime", MagicMock(side_effect=OSError("gone")))
    monkeypatch.setattr(sys, "argv", ["_child.py", "9999", "/tmp/hb.tmp", "3.0"])

    _child.main()

    mock_kill.assert_called_once_with(9999)


def test_main_kills_when_heartbeat_stale(monkeypatch):
    from litserve._win_shutdown_fix import _child

    mock_kill = MagicMock()
    monkeypatch.setattr(_child, "_alive", MagicMock(return_value=True))
    monkeypatch.setattr(_child, "_kill_subtree", mock_kill)
    monkeypatch.setattr(_child.time, "sleep", MagicMock())
    monkeypatch.setattr(_child.time, "time", MagicMock(return_value=1000.0))
    monkeypatch.setattr(_child.os.path, "getmtime", MagicMock(return_value=990.0))  # age=10 > delay=3
    monkeypatch.setattr(sys, "argv", ["_child.py", "9999", "/tmp/hb.tmp", "3.0"])

    _child.main()

    mock_kill.assert_called_once_with(9999)


def test_main_loops_until_stale(monkeypatch):
    from litserve._win_shutdown_fix import _child

    mock_sleep = MagicMock()
    mock_kill = MagicMock()
    monkeypatch.setattr(_child, "_alive", MagicMock(return_value=True))
    monkeypatch.setattr(_child, "_kill_subtree", mock_kill)
    monkeypatch.setattr(_child.time, "sleep", mock_sleep)
    # ages: 1.0, 2.0, 6.0 — kill triggered on third iteration when age > delay=3
    monkeypatch.setattr(_child.time, "time", MagicMock(side_effect=[1000.0, 1001.0, 1005.0]))
    monkeypatch.setattr(_child.os.path, "getmtime", MagicMock(return_value=999.0))
    monkeypatch.setattr(sys, "argv", ["_child.py", "9999", "/tmp/hb.tmp", "3.0"])

    _child.main()

    mock_kill.assert_called_once_with(9999)
    assert mock_sleep.call_count == 4  # initial sleep(2) + 3 × sleep(0.5)

# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Windows/PyCharm debugger shutdown sentinel. See WINDOWS_SHUTDOWN_FIX.md."""
import contextlib
import os
import sys


def _create_process_no_window(cmd_line: str) -> bool:
    # ctypes CreateProcessW bypasses pydevd's subprocess.Popen patch entirely.
    import ctypes
    from ctypes import wintypes

    class STARTUPINFOW(ctypes.Structure):
        _fields_ = (
            ("cb", wintypes.DWORD),
            ("lpReserved", wintypes.LPWSTR),
            ("lpDesktop", wintypes.LPWSTR),
            ("lpTitle", wintypes.LPWSTR),
            ("dwX", wintypes.DWORD),
            ("dwY", wintypes.DWORD),
            ("dwXSize", wintypes.DWORD),
            ("dwYSize", wintypes.DWORD),
            ("dwXCountChars", wintypes.DWORD),
            ("dwYCountChars", wintypes.DWORD),
            ("dwFillAttribute", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD),
            ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.c_void_p),
            ("hStdInput", wintypes.HANDLE),
            ("hStdOutput", wintypes.HANDLE),
            ("hStdError", wintypes.HANDLE),
        )

    class PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = (
            ("hProcess", wintypes.HANDLE),
            ("hThread", wintypes.HANDLE),
            ("dwProcessId", wintypes.DWORD),
            ("dwThreadId", wintypes.DWORD),
        )

    si = STARTUPINFOW()
    si.cb = ctypes.sizeof(STARTUPINFOW)
    pi = PROCESS_INFORMATION()
    cmd_buf = ctypes.create_unicode_buffer(cmd_line)
    ok = ctypes.windll.kernel32.CreateProcessW(
        None, cmd_buf, None, None, False, 0x08000000, None, None,
        ctypes.byref(si), ctypes.byref(pi),
    )
    if ok:
        ctypes.windll.kernel32.CloseHandle(pi.hProcess)
        ctypes.windll.kernel32.CloseHandle(pi.hThread)
    return bool(ok)


def start_heartbeat_sentinel(pid: int, heartbeat_path: str, kill_delay: float) -> None:
    """Spawn an out-of-job sentinel that kills the process tree when the heartbeat goes stale.

    Spawn chain: ctypes.CreateProcessW -> powershell.exe -File <ps1> -> WMI Win32_Process.Create
    -> python.exe _child.py. The final process runs under WmiPrvSE, outside PyCharm's Job
    Object, with no pydevd injected.
    """
    if sys.platform != "win32":
        return
    import tempfile

    child_py = os.path.join(os.path.dirname(__file__), "_child.py")
    cmd = f'"{sys.executable}" "{child_py}" {pid} "{heartbeat_path}" {kill_delay}'
    spawn_ps1 = os.path.join(tempfile.gettempdir(), f"litserve_spawn_sentinel_{pid}.ps1")
    ps1_arg = cmd.replace("'", "''")  # escape single quotes for PS single-quoted string
    with contextlib.suppress(Exception):
        with open(spawn_ps1, "w") as f:
            f.write(
                "try {\n"
                "    $r = Invoke-WmiMethod -Class Win32_Process -Name Create"
                f" -ArgumentList '{ps1_arg}'\n"
                "    exit [int]$r.ReturnValue\n"
                "} catch { exit 1 }\n"
            )

    ps_cmd = f'powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{spawn_ps1}"'
    with contextlib.suppress(Exception):
        _create_process_no_window(ps_cmd)

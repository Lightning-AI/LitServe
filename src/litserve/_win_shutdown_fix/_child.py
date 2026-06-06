"""Sentinel child process. Invoked as: python _child.py <pid> <heartbeat_path> <kill_delay>"""
import ctypes
import os
import sys
import time


def _alive(pid):
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if h:
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    return False


def _kill_subtree(root_pid):
    # Walk CreateToolhelp32Snapshot to find ALL descendants of root_pid.
    # th32ParentProcessID is fixed at creation time, so orphaned children
    # (whose parent already exited) are still found and killed.
    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong),
            ("cntUsage", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", ctypes.c_ulong),
            ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_ulong),
            ("szExeFile", ctypes.c_char * 260),
        ]

    k32 = ctypes.windll.kernel32
    snap = k32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snap == ctypes.c_void_p(-1).value or snap is None:
        return

    parent_map = {}
    pe = PROCESSENTRY32()
    pe.dwSize = ctypes.sizeof(PROCESSENTRY32)
    if k32.Process32First(snap, ctypes.byref(pe)):
        while True:
            parent_map.setdefault(pe.th32ParentProcessID, []).append(pe.th32ProcessID)
            if not k32.Process32Next(snap, ctypes.byref(pe)):
                break
    k32.CloseHandle(snap)

    descendants = []
    queue = [root_pid]
    while queue:
        cur = queue.pop()
        for child in parent_map.get(cur, []):
            if child != root_pid:
                descendants.append(child)
                queue.append(child)

    PROCESS_TERMINATE = 0x0001
    for cpid in descendants + [root_pid]:
        h = k32.OpenProcess(PROCESS_TERMINATE, False, cpid)
        if h:
            k32.TerminateProcess(h, 1)
            k32.CloseHandle(h)


def main():
    pid, hb, delay = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
    time.sleep(2)  # grace period: let server finish startup file I/O
    while True:
        time.sleep(0.5)
        if not _alive(pid):
            # Main process exited cleanly; kill any orphaned children.
            _kill_subtree(pid)
            return
        try:
            age = time.time() - os.path.getmtime(hb)
        except OSError:
            # Heartbeat file gone; assume fatal and tree-kill.
            _kill_subtree(pid)
            return
        if age > delay:
            # Heartbeat stale: main thread likely suspended by pydevd.
            _kill_subtree(pid)
            return


if __name__ == "__main__":
    main()

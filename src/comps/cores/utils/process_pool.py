# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import socket
import stat
from concurrent.futures import ProcessPoolExecutor


def close_inherited_sockets():
    """Pool initializer: fork workers inherit the server's TCP sockets and keep the connections
    it closes open. Point each at /dev/null rather than close it, so its fd number stays taken."""
    try:
        fds = [int(fd) for fd in os.listdir("/proc/self/fd")]
    except OSError:  # no /proc
        return
    devnull = os.open(os.devnull, os.O_RDWR)
    try:
        for fd in fds:
            if fd == devnull:
                continue
            try:
                if not stat.S_ISSOCK(os.fstat(fd).st_mode):
                    continue
                s = socket.socket(fileno=fd)
                family = s.family
                s.detach()
                if family in (socket.AF_INET, socket.AF_INET6):
                    os.dup2(devnull, fd)
            except OSError:  # closed meanwhile (the listdir fd itself)
                continue
    finally:
        os.close(devnull)


def new_process_pool(max_workers):
    """A ProcessPoolExecutor whose workers do not hold the server's connections."""
    return ProcessPoolExecutor(max_workers=max_workers, initializer=close_inherited_sockets)


def replace_if_broken(current, broken, max_workers):
    """The pool to use after `broken` raised BrokenProcessPool: a new one if `broken` is still
    `current`, otherwise `current` (another task already replaced it)."""
    if current is not broken:
        return current
    broken.shutdown(wait=False)
    return new_process_pool(max_workers)

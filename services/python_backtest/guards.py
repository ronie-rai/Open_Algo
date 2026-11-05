"""What a backtested script is not allowed to do, enforced inside its process.

A backtest must not be able to place a real order. The replaced
``openalgo.api`` never sends one, and these guards close the other ordinary
routes a script could take while it runs:

- **Network.** Every connection to the OpenAlgo app itself is refused at the
  socket layer, whatever library makes it: ``socket.socket.connect`` and
  ``connect_ex`` (which ``socket.create_connection``, http.client, urllib3,
  requests, httpx and asyncio's selector loops all go through), asyncio's
  Windows IOCP connect (which does not), and Unix-domain sockets (a gunicorn
  socket). "The app" is every address the configured host names resolve to at
  its port, and every local address at the app's own ports. The data loader is
  let through only while ``allow_app()`` is in force. Other hosts, such as an
  LLM provider, stay reachable.
- **Other programs.** subprocess, os.system, os.popen, os.spawn*, os.exec*,
  os.posix_spawn*, os.fork and the low-level process creators are refused.
- **The app's own code.** Importing a module that lives in the OpenAlgo
  checkout (services, database, blueprints, broker, sandbox, restx_api and the
  rest) is refused; the script's own modules beside it and installed packages
  import as usual.

These hold against every ordinary way of doing those things. They are not a
security boundary against Python code written on purpose to get around them,
which no in-process guard can be; the platform's own rule applies, as for a
live strategy: run only code you trust.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import sys
from contextlib import contextmanager
from importlib.machinery import PathFinder
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger("python_backtest")


class BlockedInBacktest(PermissionError):
    """Something a backtest must never do."""


_state = {"allow_app": False}
_blocked_pairs: set[tuple[str, int]] = set()  # (ip, port) of the app wherever it is reached
_blocked_local_ports: set[int] = set()  # the app's ports on any local address
_local_ips: set[str] = set()
_warned: set[str] = set()
_installed = {"network": False, "process": False, "imports": False}

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo


def _note(kind: str, message: str) -> None:
    if kind not in _warned:
        _warned.add(kind)
        logger.warning(message)


@contextmanager
def allow_app():
    """Let the data loader reach the app for the length of the block."""
    previous = _state["allow_app"]
    _state["allow_app"] = True
    try:
        yield
    finally:
        _state["allow_app"] = previous


# =============================================================================
# NETWORK
# =============================================================================
def _ips_of(host: str) -> set[str]:
    try:
        return {info[4][0].split("%")[0] for info in _real_getaddrinfo(host, None)}
    except (OSError, UnicodeError):
        return set()


def configure_network(urls: list[str], ports: list[int]) -> None:
    """Which addresses are the app: the hosts of `urls` at their ports, and local `ports`."""
    _blocked_pairs.clear()
    _blocked_local_ports.clear()
    _local_ips.clear()
    _local_ips.update({"127.0.0.1", "::1", "0.0.0.0", "::"})
    try:
        _local_ips.update(_ips_of(socket.gethostname()))
    except OSError:
        pass
    for port in ports:
        if port:
            _blocked_local_ports.add(int(port))
    for url in urls:
        if not url:
            continue
        parts = urlsplit(url if "://" in url else f"http://{url}")
        if not parts.hostname:
            continue
        try:
            port = parts.port or (443 if parts.scheme in ("https", "wss") else 80)
        except ValueError:
            continue
        ips = _ips_of(parts.hostname)
        for ip in ips:
            _blocked_pairs.add((ip, port))
        if parts.hostname in ("localhost",) or any(_is_local(ip) for ip in ips):
            _blocked_local_ports.add(port)


def _is_local(ip: str) -> bool:
    if ip in _local_ips:
        return True
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if getattr(addr, "ipv4_mapped", None):
        addr = addr.ipv4_mapped
    return addr.is_loopback or addr.is_unspecified or str(addr) in _local_ips


def is_blocked(address) -> bool:
    """Whether connecting to `address` would reach the app."""
    if _state["allow_app"]:
        return False
    if isinstance(address, (str, bytes)):
        return True  # a Unix-domain socket, such as gunicorn's
    if not isinstance(address, tuple) or len(address) < 2:
        return False
    host, port = str(address[0]).split("%")[0], address[1]
    try:
        port = int(port)
    except (TypeError, ValueError):
        return False
    try:
        ipaddress.ip_address(host)
        ips = {host}
    except ValueError:
        ips = _ips_of(host) or {host}
    for ip in ips:
        mapped = ip
        try:
            v6 = ipaddress.ip_address(ip)
            if getattr(v6, "ipv4_mapped", None):
                mapped = str(v6.ipv4_mapped)
        except ValueError:
            pass
        if (ip, port) in _blocked_pairs or (mapped, port) in _blocked_pairs:
            return True
        if port in _blocked_local_ports and _is_local(mapped):
            return True
    return False


def _refuse_network(address) -> None:
    _note(
        "network",
        "The strategy tried to connect to OpenAlgo directly. That is refused during a "
        "backtest, so no real order can be placed; use the client from `openalgo import api`.",
    )
    raise BlockedInBacktest(f"Connecting to OpenAlgo at {address!r} is blocked during a backtest")


def _guarded_connect(self, address):
    if is_blocked(address):
        _refuse_network(address)
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    if is_blocked(address):
        _refuse_network(address)
    return _real_connect_ex(self, address)


def install_network_guard() -> None:
    if _installed["network"]:
        return
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    # asyncio's Proactor loop (the Windows default) connects with ConnectEx and
    # never calls socket.connect.
    try:
        from asyncio import windows_events

        real_proactor_connect = windows_events.IocpProactor.connect

        def proactor_connect(self, conn, address):
            if is_blocked(address):
                _refuse_network(address)
            return real_proactor_connect(self, conn, address)

        windows_events.IocpProactor.connect = proactor_connect
    except (ImportError, AttributeError):
        pass
    _installed["network"] = True


# =============================================================================
# OTHER PROGRAMS
# =============================================================================
def _refuse_process(name: str):
    def refused(*_args, **_kwargs):
        _note(
            "process",
            f"The strategy tried to start another program ({name}). That is not allowed "
            "during a backtest.",
        )
        raise BlockedInBacktest(f"Starting another program ({name}) is blocked during a backtest")

    return refused


_OS_PROCESS_FUNCS = (
    "system", "popen", "startfile", "fork", "forkpty", "posix_spawn", "posix_spawnp",
    "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
)  # fmt: skip


def install_process_guard() -> None:
    if _installed["process"]:
        return
    # Platform details some libraries ask for are found by running a command on
    # Windows; learn them now, while that is still allowed.
    try:
        import platform

        platform.uname()
    except Exception:
        pass
    import subprocess

    subprocess.Popen._execute_child = _refuse_process("subprocess")
    for name in _OS_PROCESS_FUNCS:
        if hasattr(os, name):
            setattr(os, name, _refuse_process(f"os.{name}"))
    for module_name, attr in (("_winapi", "CreateProcess"), ("_posixsubprocess", "fork_exec")):
        try:
            module = __import__(module_name)
            if hasattr(module, attr):
                setattr(module, attr, _refuse_process(f"{module_name}.{attr}"))
        except ImportError:
            pass
    _installed["process"] = True


# =============================================================================
# THE APP'S OWN CODE
# =============================================================================
class _AppImportBlocker:
    """Refuses imports that would load a module from the OpenAlgo checkout."""

    def __init__(self, repo_root: Path, allowed_dirs: list[Path]):
        self.repo_root = repo_root.resolve()
        self.allowed = [p.resolve() for p in allowed_dirs]

    def _internal(self, location) -> bool:
        if not location:
            return False
        try:
            path = Path(location).resolve()
        except (OSError, ValueError):
            return False
        if not path.is_relative_to(self.repo_root):
            return False
        return not any(path.is_relative_to(ok) for ok in self.allowed)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in sys.modules or fullname.startswith("services.python_backtest"):
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        locations = [spec.origin, *(spec.submodule_search_locations or [])]
        if any(self._internal(loc) for loc in locations):
            _note(
                "import",
                f"The strategy tried to import {fullname} from OpenAlgo itself. A backtested "
                "script can only use the client from `openalgo import api`.",
            )
            raise ImportError(f"Importing {fullname} from OpenAlgo is blocked during a backtest")
        return None


def install_import_guard(repo_root: Path, script_dir: Path, source_dir: Path | None = None) -> None:
    """Make the strategy's imports what a live launch gives, minus the app's own code.

    A live strategy is started as ``python script.py``: its folder is first on
    the path and the checkout is not on it at all. The runner is started with
    ``-m`` from the checkout, so that entry is removed here, and a finder
    refuses anything that would still load from the checkout (a namespace
    package such as ``services``, already imported by the runner itself).

    The script runs from a copy in its run folder; ``source_dir``, the folder
    of the live script, comes next on the path so a helper module kept beside
    the live script still imports.
    """
    if _installed["imports"]:
        return
    root = repo_root.resolve()

    def is_root(entry: str) -> bool:
        try:
            return Path(entry or os.getcwd()).resolve() == root
        except (OSError, ValueError):
            return False

    sys.path[:] = [p for p in sys.path if not is_root(p)]
    front = [script_dir] + ([source_dir] if source_dir else [])
    sys.path[0:0] = [str(d) for d in front]
    allowed = [*front, Path(sys.prefix), Path(sys.base_prefix)]
    sys.meta_path.insert(0, _AppImportBlocker(root, allowed))
    sys.path_importer_cache.clear()
    _installed["imports"] = True

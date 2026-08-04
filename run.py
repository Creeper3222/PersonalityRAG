from __future__ import annotations

import os
import socket
import sys
import webbrowser
import asyncio
from pathlib import Path

import uvicorn

import personalityrag.app as app_module
from personalityrag.config import build_access_url
from personalityrag.instance_lock import InstanceLock, SingleInstanceError
from personalityrag.listener_surface import (
    ADAPTER_ACCESS_SURFACE,
    WEBUI_SURFACE,
    listener_app,
)
from personalityrag.version import display_version


ROOT = Path(__file__).resolve().parent
config = app_module.config


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def resolve_port(
    host: str,
    preferred_port: int,
    scan_limit: int = 80,
    *,
    exclude_ports: set[int] | None = None,
    label: str = "配置端口",
) -> tuple[int, str]:
    exclude_ports = exclude_ports or set()
    if preferred_port not in exclude_ports and port_available(host, preferred_port):
        return preferred_port, ""
    for port in range(preferred_port + 1, preferred_port + scan_limit + 1):
        if port in exclude_ports:
            continue
        if port_available(host, port):
            return port, (
                f"{label} {host}:{preferred_port} 已被占用或已被本进程其它服务使用，"
                f"本次启动自动回退到 {host}:{port}；配置文件不会被改写。"
            )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        fallback = int(sock.getsockname()[1])
    return fallback, (
        f"{label} {host}:{preferred_port} 及后续 {scan_limit} 个端口均不可用，"
        f"本次启动自动回退到系统分配端口 {host}:{fallback}；配置文件不会被改写。"
    )


async def serve_dual_ports(host: str, webui_port: int, access_port: int) -> None:
    webui_app = listener_app(app_module.app, WEBUI_SURFACE)
    access_app = listener_app(app_module.app, ADAPTER_ACCESS_SURFACE)
    webui_server = uvicorn.Server(
        uvicorn.Config(
            webui_app,
            host=host,
            port=webui_port,
            reload=False,
            log_level="info",
            access_log=False,
            server_header=False,
            timeout_keep_alive=15,
            timeout_graceful_shutdown=60,
            backlog=128,
        )
    )
    access_server = uvicorn.Server(
        uvicorn.Config(
            access_app,
            host=host,
            port=access_port,
            reload=False,
            log_level="info",
            access_log=False,
            lifespan="off",
            server_header=False,
            timeout_keep_alive=15,
            timeout_graceful_shutdown=60,
            backlog=128,
        )
    )
    def request_shutdown() -> None:
        webui_server.should_exit = True
        access_server.should_exit = True

    app_module.set_process_shutdown_callback(request_shutdown)
    webui_task = asyncio.create_task(webui_server.serve())
    try:
        for _ in range(200):
            if webui_server.started or webui_task.done():
                break
            await asyncio.sleep(0.05)
        if webui_task.done():
            webui_task.result()
        access_task = asyncio.create_task(access_server.serve())
        done, pending = await asyncio.wait(
            {webui_task, access_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
    finally:
        app_module.set_process_shutdown_callback(None)


if __name__ == "__main__":
    lock = InstanceLock(ROOT / "data" / "logs" / "personalityrag.instance.lock")
    clean_shutdown = False
    try:
        lock.acquire()
    except SingleInstanceError as exc:
        print(f"[PersonalityRAG] 启动被拒绝：{exc}", file=sys.stderr)
        raise SystemExit(3) from exc

    try:
        actual_port, webui_warning = resolve_port(
            config.host,
            config.port,
            label="WebUI 配置端口",
        )
        actual_access_port, access_warning = resolve_port(
            config.host,
            config.access_port,
            exclude_ports={actual_port},
            label="记忆库接入配置端口",
        )
        os.environ["PERSONALITYRAG_ACTUAL_PORT"] = str(actual_port)
        os.environ["PERSONALITYRAG_ACCESS_ACTUAL_PORT"] = str(actual_access_port)
        if webui_warning:
            os.environ["PERSONALITYRAG_WEBUI_PORT_FALLBACK_WARNING"] = webui_warning
            print(f"[PersonalityRAG] WARNING: {webui_warning}")
        if access_warning:
            os.environ["PERSONALITYRAG_ACCESS_PORT_FALLBACK_WARNING"] = access_warning
            print(f"[PersonalityRAG] WARNING: {access_warning}")

        url = build_access_url(config.access_base_url, actual_port)
        api_url = build_access_url(config.access_base_url, actual_access_port)
        print(f"[PersonalityRAG] {display_version()}")
        print(f"[PersonalityRAG] WebUI: {url}")
        print(f"[PersonalityRAG] 记忆库接入: {api_url}")
        if config.webui_password_hash:
            print("[PersonalityRAG] WebUI 登录：使用已设置的登录密码。")
            print("[PersonalityRAG] API key：仍可作为 Bearer Token 用于脚本/API 访问。")
        else:
            print(
                "[PersonalityRAG] API key fingerprint: "
                + config.api_key_fingerprint
                + "（首次登录密钥保存在 config/config.json；请妥善保管）"
            )
        if os.environ.get("PERSONALITYRAG_SUPPRESS_BROWSER") != "1":
            webbrowser.open(url)
        asyncio.run(serve_dual_ports(config.host, actual_port, actual_access_port))
        clean_shutdown = True
    finally:
        lock.release()
    if clean_shutdown:
        # The ASGI lifespan has already drained jobs, runtimes, SQLite pools and
        # HTTP transports, and the single-instance lock is released above.  A
        # third-party/native worker that survives Python interpreter finalizers
        # must not leave a portless zombie behind or accumulate on each WebUI
        # restart.  Exit only after the full graceful path has completed; real
        # startup or runtime exceptions still propagate normally.
        os._exit(0)

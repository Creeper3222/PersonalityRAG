from __future__ import annotations

import os
import socket
import sys
import webbrowser
from pathlib import Path

import uvicorn

from personalityrag.config import load_config
from personalityrag.instance_lock import InstanceLock, SingleInstanceError


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "config.json"
config = load_config(CONFIG_PATH)


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def resolve_port(host: str, preferred_port: int, scan_limit: int = 80) -> tuple[int, str]:
    if port_available(host, preferred_port):
        return preferred_port, ""
    for port in range(preferred_port + 1, preferred_port + scan_limit + 1):
        if port_available(host, port):
            return port, (
                f"配置端口 {host}:{preferred_port} 已被占用，"
                f"本次启动自动回退到 {host}:{port}；配置文件不会被改写。"
            )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        fallback = int(sock.getsockname()[1])
    return fallback, (
        f"配置端口 {host}:{preferred_port} 及后续 {scan_limit} 个端口均不可用，"
        f"本次启动自动回退到系统分配端口 {host}:{fallback}；配置文件不会被改写。"
    )


if __name__ == "__main__":
    lock = InstanceLock(ROOT / "data" / "logs" / "personalityrag.instance.lock")
    try:
        lock.acquire()
    except SingleInstanceError as exc:
        print(f"[PersonalityRAG] 启动被拒绝：{exc}", file=sys.stderr)
        raise SystemExit(3) from exc

    try:
        actual_port, fallback_warning = resolve_port(config.host, config.port)
        os.environ["PERSONALITYRAG_ACTUAL_PORT"] = str(actual_port)
        if fallback_warning:
            os.environ["PERSONALITYRAG_PORT_FALLBACK_WARNING"] = fallback_warning
            print(f"[PersonalityRAG] WARNING: {fallback_warning}")

        url = f"http://{config.host}:{actual_port}/"
        print("[PersonalityRAG] v0.1.0")
        print(f"[PersonalityRAG] WebUI: {url}")
        if config.webui_password_hash:
            print("[PersonalityRAG] WebUI 登录：使用已设置的登录密码。")
            print("[PersonalityRAG] API key：仍可作为 Bearer Token 用于脚本/API 访问。")
        else:
            print(
                "[PersonalityRAG] API key: "
                + config.api_key
                + "（首次登录使用；请妥善保存）"
            )
        webbrowser.open(url)
        uvicorn.run(
            "personalityrag.app:app",
            host=config.host,
            port=actual_port,
            reload=False,
            log_level="info",
        )
    finally:
        lock.release()

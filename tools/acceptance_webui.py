from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import websockets


ROOT = Path(__file__).resolve().parents[1]


def default_state_root() -> Path:
    return Path(os.environ.get("PERSONALITYRAG_STATE_ROOT", ROOT)).resolve()


def default_credential(state_root: Path) -> str:
    config_path = state_root / "config" / "config.json"
    if not config_path.exists():
        return ""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("login_password_hash"):
        return ""
    return str(config.get("api_key") or "")


def default_api_key(state_root: Path) -> str:
    config_path = state_root / "config" / "config.json"
    if not config_path.exists():
        return ""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return str(config.get("api_key") or "")


async def run(
    devtools_url: str,
    *,
    base_url: str,
    credential: str,
    report_dir: Path,
    api_key: str = "",
) -> dict:
    base_url = base_url.rstrip("/")
    pages = json.loads(
        urllib.request.urlopen(f"{devtools_url}/json/list", timeout=5).read()
    )
    page = next(
        (item for item in pages if item.get("type") == "page"),
        pages[0],
    )
    async with websockets.connect(
        page["webSocketDebuggerUrl"], max_size=20_000_000
    ) as ws:
        counter = 0
        errors = []

        async def call(method: str, params: dict | None = None):
            nonlocal counter
            counter += 1
            request_id = counter
            await ws.send(
                json.dumps(
                    {
                        "id": request_id,
                        "method": method,
                        "params": params or {},
                    }
                )
            )
            while True:
                payload = json.loads(await ws.recv())
                if payload.get("method") == "Runtime.exceptionThrown":
                    errors.append(payload)
                if payload.get("id") == request_id:
                    if "error" in payload:
                        raise RuntimeError(payload["error"])
                    return payload.get("result", {})

        await call("Runtime.enable")
        await call("Page.enable")
        if api_key:
            await call("Network.enable")
            await call(
                "Network.setExtraHTTPHeaders",
                {"headers": {"Authorization": f"Bearer {api_key}"}},
            )
        await call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 1600,
                "height": 1000,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
        await call("Page.navigate", {"url": f"{base_url}/"})
        await asyncio.sleep(1.5)
        if credential:
            expression = (
                "fetch('/api/v1/auth/login',"
                "{method:'POST',headers:{'Content-Type':'application/json'},"
                f"body:JSON.stringify({{credential:{json.dumps(credential)}}})}})"
                ".then(async r=>{if(!r.ok)throw new Error(await r.text());return r.json()})"
            )
            await call(
                "Runtime.evaluate",
                {
                    "expression": expression,
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            )
        await call("Page.navigate", {"url": f"{base_url}/"})
        await asyncio.sleep(2.5)
        libraries = await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "({title:document.querySelector('#page-title')?.textContent,"
                    "cards:document.querySelectorAll("
                    "'#library-cards .management-card').length,"
                    "body:document.body.innerText.slice(0,1200),"
                    "selectedLibraryId:localStorage.getItem('prag_library_id')||''})"
                ),
                "returnByValue": True,
            },
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        screenshot = await call(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False},
        )
        (report_dir / "multilibrary-ui.png").write_bytes(
            base64.b64decode(screenshot["data"])
        )
        async def capture_library_key(theme: str, filename: str) -> dict:
            current_theme = await call(
                "Runtime.evaluate",
                {
                    "expression": "document.documentElement.dataset.theme",
                    "returnByValue": True,
                },
            )
            if current_theme["result"]["value"] != theme:
                await call(
                    "Runtime.evaluate",
                    {"expression": "document.querySelector('#theme-toggle').click()"},
                )
                await asyncio.sleep(0.2)
            style = await call(
                "Runtime.evaluate",
                {
                    "expression": (
                        "(()=>{const el=document.querySelector('.key-library-fab');"
                        "if(!el)return {visible:false};const s=getComputedStyle(el);"
                        "const root=getComputedStyle(document.documentElement);return {"
                        "visible:el.getBoundingClientRect().width>0,"
                        "inViewport:(()=>{const r=el.getBoundingClientRect();return "
                        "r.top>=0&&r.left>=0&&r.bottom<=innerHeight&&r.right<=innerWidth})(),"
                        "color:s.color,accent:root.getPropertyValue('--accent').trim(),"
                        "overflow:document.documentElement.scrollWidth>innerWidth}})()"
                    ),
                    "returnByValue": True,
                },
            )
            image = await call(
                "Page.captureScreenshot",
                {"format": "png", "captureBeyondViewport": False},
            )
            (report_dir / filename).write_bytes(base64.b64decode(image["data"]))
            return style["result"]["value"]

        library_key_light = await capture_library_key(
            "light", "library-key-light-desktop.png"
        )
        library_key_dark = await capture_library_key(
            "dark", "library-key-dark-desktop.png"
        )
        await call(
            "Emulation.setDeviceMetricsOverride",
            {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True},
        )
        await call(
            "Runtime.evaluate",
            {"expression": "document.querySelector('.key-library-fab').scrollIntoView({block:'center'})"},
        )
        await asyncio.sleep(0.2)
        library_key_mobile = await capture_library_key(
            "dark", "library-key-dark-mobile.png"
        )
        await call(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1600, "height": 1000, "deviceScaleFactor": 1, "mobile": False},
        )
        await capture_library_key("light", "library-key-light-restored.png")
        await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "document.querySelector("
                    "'.nav[data-page=providers]').click()"
                )
            },
        )
        await asyncio.sleep(2)
        providers = await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "({title:document.querySelector('#page-title')?.textContent,"
                    "cards:document.querySelectorAll("
                    "'#provider-cards .management-card').length,"
                    "body:document.body.innerText.slice(0,1200)})"
                ),
                "returnByValue": True,
            },
        )
        screenshot = await call(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False},
        )
        (report_dir / "providers-ui.png").write_bytes(
            base64.b64decode(screenshot["data"])
        )
        await call(
            "Runtime.evaluate",
            {"expression": "document.querySelector('#provider-create').click()"},
        )
        await asyncio.sleep(0.3)
        type_picker = await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "({cards:document.querySelectorAll("
                    "'#provider-type-cards .provider-type-card').length})"
                ),
                "returnByValue": True,
            },
        )
        await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "document.querySelector("
                    "'#provider-type-cards [data-type=vllm_embedding]').click()"
                )
            },
        )
        await asyncio.sleep(0.3)
        editor = await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "({fields:['provider-id','provider-name','provider-enabled',"
                    "'provider-api-key','provider-api-base','provider-model',"
                    "'provider-dimensions','provider-timeout','provider-proxy',"
                    "'provider-batch','provider-concurrency','provider-retries']"
                    ".filter(id=>document.getElementById(id)).length,"
                    "createIdEditable:!document.querySelector('#provider-id').readOnly})"
                ),
                "returnByValue": True,
            },
        )
        editor_existing = {"result": {"value": {
            "idReadOnly": True,
            "secretBlank": True,
            "clearVisible": False,
            "existingProviderPresent": False,
        }}}
        existing_provider = await call(
            "Runtime.evaluate",
            {
                "expression": "Boolean(document.querySelector('.edit-provider'))",
                "returnByValue": True,
            },
        )
        if existing_provider["result"]["value"]:
            await call(
                "Runtime.evaluate",
                {
                    "expression": (
                        "document.querySelector('#provider-modal').classList.add('hidden');"
                        "document.querySelector('.edit-provider').click()"
                    )
                },
            )
            await asyncio.sleep(0.3)
            editor_existing = await call(
                "Runtime.evaluate",
                {
                    "expression": (
                        "({idReadOnly:document.querySelector('#provider-id').readOnly,"
                        "secretBlank:document.querySelector('#provider-api-key').value==='',"
                        "clearVisible:!document.querySelector("
                        "'#provider-clear-key-row').classList.contains('hidden'),"
                        "existingProviderPresent:true})"
                    ),
                    "returnByValue": True,
                },
            )
        await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "document.querySelector('#provider-modal').classList.add('hidden')"
                )
            },
        )
        locale_results = {}
        for language in ("en", "ru", "zh"):
            await call(
                "Runtime.evaluate",
                {
                    "expression": (
                        "document.querySelector('#language').value="
                        + json.dumps(language)
                        + ";document.querySelector('#language')"
                        ".dispatchEvent(new Event('change',{bubbles:true}))"
                    )
                },
            )
            await asyncio.sleep(0.7)
            localized = await call(
                "Runtime.evaluate",
                {
                    "expression": (
                        "({title:document.querySelector('#page-title')?.textContent,"
                        "create:document.querySelector('#provider-create')?.textContent,"
                        "card:document.querySelector('#provider-cards')?.innerText.slice(0,500)})"
                    ),
                    "returnByValue": True,
                },
            )
            locale_results[language] = localized["result"]["value"]
        before_theme = await call(
            "Runtime.evaluate",
            {
                "expression": "document.documentElement.dataset.theme",
                "returnByValue": True,
            },
        )
        await call(
            "Runtime.evaluate",
            {"expression": "document.querySelector('#theme-toggle').click()"},
        )
        after_theme = await call(
            "Runtime.evaluate",
            {
                "expression": "document.documentElement.dataset.theme",
                "returnByValue": True,
            },
        )
        await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "document.querySelector('.nav[data-page=files]').click()"
                )
            },
        )
        await asyncio.sleep(1.2)
        files = await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "({title:document.querySelector('#page-title')?.textContent||'',"
                    "rows:document.querySelectorAll('#file-table-body tr').length,"
                    "protectedItems:document.querySelectorAll('.file-badge-readonly').length,"
                    "folderIcons:document.querySelectorAll('.file-icon--folder').length,"
                    "textIcons:document.querySelectorAll('.file-icon--file-text').length,"
                    "codeIcons:document.querySelectorAll('.file-icon--file-code').length,"
                    "genericIcons:document.querySelectorAll('.file-icon--file-generic').length,"
                    "visibleInternal:Array.from(document.querySelectorAll('.file-name-copy strong'))"
                    ".some(node=>['.git','.venv','.pytest_cache','.ruff_cache'].includes(node.textContent))})"
                ),
                "returnByValue": True,
            },
        )
        await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "Array.from(document.querySelectorAll('#file-table-body tr'))"
                    ".find(row=>row.querySelector('.file-name-copy strong')?.textContent==='run.py')"
                    "?.querySelector('[data-file-open]')?.click()"
                )
            },
        )
        await asyncio.sleep(0.5)
        file_preview = await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "({visible:!document.querySelector('#file-preview-modal')?.classList.contains('hidden'),"
                    "contentLength:document.querySelector('#file-preview-content')?.textContent.length||0})"
                ),
                "returnByValue": True,
            },
        )
        await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "document.querySelector('[data-file-modal-close=file-preview-modal]')?.click();"
                    "Array.from(document.querySelectorAll('#file-table-body tr'))"
                    ".find(row=>row.querySelector('.file-name-copy strong')?.textContent==='docs')"
                    "?.querySelector('[data-file-open]')?.click()"
                )
            },
        )
        await asyncio.sleep(0.5)
        file_navigation = await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "({path:document.querySelector('#file-current-path')?.textContent||'',"
                    "breadcrumbCount:document.querySelectorAll('[data-file-breadcrumb]').length})"
                ),
                "returnByValue": True,
            },
        )
        await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "document.querySelector('[data-file-breadcrumb=\"\"]')?.click()"
                )
            },
        )
        await asyncio.sleep(0.5)
        await call(
            "Runtime.evaluate",
            {"expression": "document.documentElement.dataset.theme='light'"},
        )
        screenshot = await call(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False},
        )
        (report_dir / "files-light-desktop.png").write_bytes(
            base64.b64decode(screenshot["data"])
        )
        await call(
            "Runtime.evaluate",
            {"expression": "document.documentElement.dataset.theme='dark'"},
        )
        screenshot = await call(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False},
        )
        (report_dir / "files-dark-desktop.png").write_bytes(
            base64.b64decode(screenshot["data"])
        )
        await call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 430,
                "height": 900,
                "deviceScaleFactor": 1,
                "mobile": True,
            },
        )
        screenshot = await call(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False},
        )
        (report_dir / "files-dark-mobile.png").write_bytes(
            base64.b64decode(screenshot["data"])
        )
        await call(
            "Runtime.evaluate",
            {"expression": "document.documentElement.dataset.theme='light'"},
        )
        screenshot = await call(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False},
        )
        (report_dir / "files-light-mobile.png").write_bytes(
            base64.b64decode(screenshot["data"])
        )
        await call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 1600,
                "height": 1000,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
        page_smoke = {}
        for page_name in (
            "libraries",
            "providers",
            "graph",
            "memory",
            "recall",
            "system",
            "files",
            "settings",
            "logs",
        ):
            await call(
                "Runtime.evaluate",
                {
                    "expression": (
                        "document.querySelector("
                        + json.dumps(f'.nav[data-page="{page_name}"]')
                        + ")?.click()"
                    )
                },
            )
            await asyncio.sleep(0.5)
            snapshot = await call(
                "Runtime.evaluate",
                {
                    "expression": (
                        "({title:document.querySelector('#page-title')?.textContent||'',"
                        "appVisible:!document.querySelector('#app')?.classList.contains('hidden'),"
                        "pageVisible:document.querySelector("
                        + json.dumps(f"#page-{page_name}")
                        + ")?.classList.contains('active')===true})"
                    ),
                    "returnByValue": True,
                },
            )
            page_smoke[page_name] = snapshot["result"]["value"]
        return {
            "libraries": libraries["result"]["value"],
            "library_key": {
                "light": library_key_light,
                "dark": library_key_dark,
                "mobile": library_key_mobile,
            },
            "providers": providers["result"]["value"],
            "provider_editor": {
                "type_cards": type_picker["result"]["value"]["cards"],
                **editor["result"]["value"],
                **editor_existing["result"]["value"],
            },
            "locales": locale_results,
            "theme_changed": (
                before_theme["result"]["value"]
                != after_theme["result"]["value"]
            ),
            "files": {
                **files["result"]["value"],
                "preview": file_preview["result"]["value"],
                "navigation": file_navigation["result"]["value"],
            },
            "page_smoke": page_smoke,
            "runtime_exceptions": len(errors),
            "runtime_exception_messages": [
                item.get("params", {})
                .get("exceptionDetails", {})
                .get("exception", {})
                .get("description", "unknown runtime exception")
                for item in errors
            ],
        }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devtools-url", default="http://127.0.0.1:9222"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--state-root", type=Path, default=default_state_root())
    parser.add_argument("--credential", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "personalityrag-webui-acceptance",
    )
    args = parser.parse_args()
    credential = args.credential or default_credential(args.state_root.resolve())
    api_key = args.api_key or default_api_key(args.state_root.resolve())
    if not credential and not api_key:
        parser.error(
            "--credential or --api-key is required for WebUI authentication"
        )
    report_dir = args.report_dir.resolve()
    result = asyncio.run(
        run(
            args.devtools_url,
            base_url=args.base_url,
            credential=credential,
            api_key=api_key,
            report_dir=report_dir,
        )
    )
    report_path = report_dir / "webui-acceptance.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if (
        result["runtime_exceptions"] == 0
        and result["libraries"]["cards"] >= 1
        and all(
            item["visible"] and item["inViewport"] and not item["overflow"]
            for item in result["library_key"].values()
        )
        and result["providers"]["cards"] >= 1
        and result["provider_editor"]["type_cards"] >= 3
        and result["provider_editor"]["fields"] == 12
        and result["provider_editor"]["createIdEditable"]
        and (
            not result["provider_editor"]["existingProviderPresent"]
            or (
                result["provider_editor"]["idReadOnly"]
                and result["provider_editor"]["secretBlank"]
            )
        )
        and result["locales"]["zh"]["title"] == "模型提供商"
        and result["locales"]["en"]["title"] == "Providers"
        and result["locales"]["ru"]["title"] == "Провайдеры"
        and result["theme_changed"]
        and result["files"]["rows"] >= 1
        and result["files"]["protectedItems"] >= 1
        and result["files"]["folderIcons"] >= 1
        and result["files"]["codeIcons"] >= 1
        and not result["files"]["visibleInternal"]
        and result["files"]["preview"]["visible"]
        and result["files"]["preview"]["contentLength"] > 0
        and result["files"]["navigation"]["path"].endswith("/docs")
        and result["files"]["navigation"]["breadcrumbCount"] >= 2
        and all(
            item["appVisible"] and item["pageVisible"]
            for item in result["page_smoke"].values()
        )
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())

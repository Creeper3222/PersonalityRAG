from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import urllib.request
from pathlib import Path

import websockets


ROOT = Path(__file__).resolve().parents[1]


async def run(devtools_url: str) -> dict:
    config = json.loads(
        (ROOT / "config" / "config.json").read_text(encoding="utf-8")
    )
    pages = json.loads(
        urllib.request.urlopen(f"{devtools_url}/json/list").read()
    )
    page = pages[0]
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
        await call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 1600,
                "height": 1000,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
        await call("Page.navigate", {"url": "http://127.0.0.1:8765/"})
        await asyncio.sleep(1.5)
        expression = (
            "fetch('/api/v1/auth/login',"
            "{method:'POST',headers:{'Content-Type':'application/json'},"
            f"body:JSON.stringify({{api_key:{json.dumps(config['api_key'])}}})}})"
            ".then(r=>r.json())"
        )
        await call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
            },
        )
        await call("Page.navigate", {"url": "http://127.0.0.1:8765/"})
        await asyncio.sleep(2.5)
        libraries = await call(
            "Runtime.evaluate",
            {
                "expression": (
                    "({title:document.querySelector('#page-title')?.textContent,"
                    "cards:document.querySelectorAll("
                    "'#library-cards .management-card').length,"
                    "body:document.body.innerText.slice(0,1200)})"
                ),
                "returnByValue": True,
            },
        )
        report_dir = (
            ROOT / "data" / "libraries" / "beileite" / "reports"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        screenshot = await call(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False},
        )
        (report_dir / "multilibrary-ui.png").write_bytes(
            base64.b64decode(screenshot["data"])
        )
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
                    "'#provider-clear-key-row').classList.contains('hidden')})"
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
        return {
            "libraries": libraries["result"]["value"],
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
            "runtime_exceptions": len(errors),
        }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devtools-url", default="http://127.0.0.1:9222"
    )
    args = parser.parse_args()
    result = asyncio.run(run(args.devtools_url))
    report_path = (
        ROOT
        / "data"
        / "libraries"
        / "beileite"
        / "reports"
        / "webui-acceptance.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if (
        result["runtime_exceptions"] == 0
        and result["libraries"]["cards"] >= 1
        and result["providers"]["cards"] >= 1
        and result["provider_editor"]["type_cards"] == 3
        and result["provider_editor"]["fields"] == 12
        and result["provider_editor"]["createIdEditable"]
        and result["provider_editor"]["idReadOnly"]
        and result["provider_editor"]["secretBlank"]
        and result["locales"]["zh"]["title"] == "模型提供商"
        and result["locales"]["en"]["title"] == "Providers"
        and result["locales"]["ru"]["title"] == "Провайдеры"
        and result["theme_changed"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())

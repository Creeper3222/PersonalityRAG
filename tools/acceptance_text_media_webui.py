from __future__ import annotations

import argparse
import asyncio
import base64
import json
import urllib.request
from pathlib import Path

import websockets


async def run(
    devtools_url: str,
    *,
    base_url: str,
    api_key: str,
    report_dir: Path,
) -> dict:
    pages = json.loads(
        urllib.request.urlopen(f"{devtools_url}/json/list", timeout=5).read()
    )
    page = next((item for item in pages if item.get("type") == "page"), pages[0])
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=20_000_000) as ws:
        counter = 0
        exceptions: list[dict] = []

        async def call(method: str, params: dict | None = None) -> dict:
            nonlocal counter
            counter += 1
            request_id = counter
            await ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
            while True:
                payload = json.loads(await ws.recv())
                if payload.get("method") == "Runtime.exceptionThrown":
                    exceptions.append(payload)
                if payload.get("id") == request_id:
                    if "error" in payload:
                        raise RuntimeError(payload["error"])
                    return payload.get("result", {})

        async def evaluate(expression: str):
            result = await call(
                "Runtime.evaluate",
                {"expression": expression, "awaitPromise": True, "returnByValue": True},
            )
            return result.get("result", {}).get("value")

        async def screenshot(name: str) -> None:
            payload = await call(
                "Page.captureScreenshot",
                {"format": "png", "captureBeyondViewport": False},
            )
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / name).write_bytes(base64.b64decode(payload["data"]))

        await call("Runtime.enable")
        await call("Page.enable")
        await call("Network.enable")
        await call(
            "Network.setExtraHTTPHeaders",
            {"headers": {"Authorization": f"Bearer {api_key}"}},
        )
        await call(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1600, "height": 1000, "deviceScaleFactor": 1, "mobile": False},
        )
        await call("Page.navigate", {"url": base_url.rstrip("/") + "/"})
        await asyncio.sleep(2.2)
        await evaluate(
            "document.querySelector('[data-database-category=knowledge]').click()"
        )
        await asyncio.sleep(0.5)
        library_page = await evaluate(
            "({cards:document.querySelectorAll('.text-media-library-card').length,"
            "type:document.querySelector('.text-media-library-card .library-card-summary-meta dd')?.textContent||'',"
            "overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth})"
        )
        await screenshot("text-media-databases-desktop.png")

        await evaluate("document.querySelector('.manage-text-media').click()")
        await asyncio.sleep(0.8)
        content = await evaluate(
            "({visible:document.querySelector('#page-database').classList.contains('active'),"
            "tabs:document.querySelectorAll('#database-type-nav [data-route^=\"database:text_media_v1:\"]').length,"
            "overflow:document.querySelector('#page-database').scrollWidth>document.querySelector('#page-database').clientWidth})"
        )
        await screenshot("text-media-content-desktop.png")

        await evaluate("document.querySelector('[data-route=\"database:text_media_v1:media\"]').click()")
        await asyncio.sleep(0.25)
        await screenshot("text-media-media-desktop.png")
        await evaluate("document.querySelector('[data-route=\"database:text_media_v1:search\"]').click()")
        await asyncio.sleep(0.25)
        await screenshot("text-media-search-desktop.png")
        await evaluate("document.querySelector('[data-route=\"database:text_media_v1:settings\"]').click()")
        await asyncio.sleep(0.25)
        settings = await evaluate(
            "({warning:document.querySelector('[data-i18n=tmkbPlaintextWarning]')?.textContent||'',"
            "importField:Boolean(document.querySelector('#text-media-import-file')),"
            "exportButton:Boolean(document.querySelector('#text-media-export'))})"
        )
        await screenshot("text-media-settings-desktop.png")

        await call(
            "Emulation.setDeviceMetricsOverride",
            {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True},
        )
        await asyncio.sleep(0.4)
        mobile = await evaluate(
            "({viewport:window.innerWidth,pageWidth:document.querySelector('#page-database').getBoundingClientRect().width,"
            "bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth,"
            "pageOverflow:document.querySelector('#page-database').scrollWidth>document.querySelector('#page-database').clientWidth})"
        )
        await screenshot("text-media-settings-mobile.png")

        await evaluate("document.querySelector('[data-page=libraries]').click()")
        await evaluate("document.querySelector('#library-create').click()")
        await asyncio.sleep(0.25)
        type_picker = await evaluate(
            "({cards:document.querySelectorAll('#database-type-cards .database-type-card').length,"
            "text:document.querySelector('#database-type-cards')?.innerText||''})"
        )
        await screenshot("text-media-type-picker-mobile.png")
        await evaluate("document.querySelector('#database-type-cards [data-type=text_media_v1]').click()")
        await asyncio.sleep(0.25)
        create_modal = await evaluate(
            "({visible:!document.querySelector('#text-media-create-modal').classList.contains('hidden'),"
            "providerOptions:document.querySelectorAll('#text-media-create-provider option').length,"
            "overflow:document.querySelector('.text-media-create-modal').scrollWidth>document.querySelector('.text-media-create-modal').clientWidth})"
        )
        await screenshot("text-media-create-mobile.png")

        return {
            "library_page": library_page,
            "content": content,
            "settings": settings,
            "mobile": mobile,
            "type_picker": type_picker,
            "create_modal": create_modal,
            "browser_exceptions": len(exceptions),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devtools-url", default="http://127.0.0.1:9225")
    parser.add_argument("--base-url", default="http://127.0.0.1:8877")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(
        run(
            args.devtools_url,
            base_url=args.base_url,
            api_key=args.api_key,
            report_dir=args.report_dir,
        )
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

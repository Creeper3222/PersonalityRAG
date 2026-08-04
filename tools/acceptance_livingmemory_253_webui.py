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
    database_id: str,
    report_dir: Path,
) -> dict[str, object]:
    pages = json.loads(
        urllib.request.urlopen(f"{devtools_url}/json/list", timeout=5).read()
    )
    page = next((item for item in pages if item.get("type") == "page"), pages[0])
    async with websockets.connect(
        page["webSocketDebuggerUrl"],
        max_size=20_000_000,
    ) as ws:
        counter = 0
        exceptions: list[dict[str, object]] = []

        async def call(method: str, params: dict | None = None) -> dict:
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
                    exceptions.append(payload)
                if payload.get("id") == request_id:
                    if "error" in payload:
                        raise RuntimeError(payload["error"])
                    return payload.get("result", {})

        async def evaluate(expression: str):
            result = await call(
                "Runtime.evaluate",
                {
                    "expression": expression,
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            )
            return result.get("result", {}).get("value")

        async def wait_for(expression: str, *, timeout: float = 8.0):
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                value = await evaluate(expression)
                if value:
                    return value
                await asyncio.sleep(0.12)
            raise TimeoutError(f"browser condition timed out: {expression}")

        async def screenshot(name: str) -> None:
            payload = await call(
                "Page.captureScreenshot",
                {"format": "png", "captureBeyondViewport": False},
            )
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / name).write_bytes(base64.b64decode(payload["data"]))

        async def set_theme(theme: str) -> None:
            await evaluate(
                "(()=>{const target="
                + json.dumps(theme)
                + ";if(document.documentElement.dataset.theme!==target){"
                "document.querySelector('#theme-toggle')?.click();}"
                "return document.documentElement.dataset.theme===target})()"
            )
            await asyncio.sleep(0.15)

        async def open_database_page(page_id: str) -> None:
            route = f"database:livingmemory_v8:{page_id}"
            clicked = await evaluate(
                "(()=>{const button=document.querySelector("
                + json.dumps(f'[data-route="{route}"]')
                + ");if(!button)return false;button.click();return true})()"
            )
            if not clicked:
                raise RuntimeError(f"database navigation is missing: {route}")
            await wait_for(
                "document.querySelector("
                + json.dumps(f'[data-route="{route}"]')
                + ")?.classList.contains('active')"
            )

        await call("Runtime.enable")
        await call("Page.enable")
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
        await call("Page.navigate", {"url": base_url.rstrip("/") + "/"})
        await wait_for(
            "document.querySelectorAll('.library-card').length>0",
            timeout=12,
        )

        selected = await evaluate(
            "(()=>{const card=[...document.querySelectorAll('.library-card')].find("
            f"item=>item.dataset.id==={json.dumps(database_id)}"
            ");if(!card)return false;"
            "const button=card.querySelector('.enter-library');"
            "if(!button)return false;button.click();return true})()"
        )
        if not selected:
            raise RuntimeError(f"LivingMemory acceptance database not found: {database_id}")
        await wait_for(
            "Boolean(document.querySelector("
            "'[data-route=\"database:livingmemory_v8:memories\"]'))"
        )

        await open_database_page("memories")
        await wait_for("document.querySelectorAll('#memory-rows .memory-row').length>0")
        await evaluate(
            "(()=>{const select=document.querySelector('#memory-status');"
            "select.value='active';"
            "select.dispatchEvent(new Event('change',{bubbles:true}));return true})()"
        )
        await wait_for("document.querySelectorAll('#memory-rows .memory-row').length>1")
        await set_theme("light")
        memory_desktop = await evaluate(
            "({"
            "rows:document.querySelectorAll('#memory-rows .memory-row').length,"
            "transfer:Boolean(document.querySelector('.memory-transfer-panel')),"
            "bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth,"
            "pageOverflow:document.querySelector('#page-memory').scrollWidth>"
            "document.querySelector('#page-memory').clientWidth"
            "})"
        )
        await screenshot("livingmemory-253-memories-light-desktop.png")
        await set_theme("dark")
        await screenshot("livingmemory-253-memories-dark-desktop.png")

        await evaluate("document.querySelector('#memory-rows .memory-row').click()")
        await wait_for(
            "document.querySelector('#memory-detail-panel')?.classList.contains('visible')"
            "&&Boolean(document.querySelector('#memory-detail-source'))"
            "&&document.querySelector('#memory-detail-panel').getBoundingClientRect().right"
            "<=window.innerWidth+1"
        )
        detail = await evaluate(
            "({"
            "canonical:Boolean(document.querySelector('.memory-detail-content')),"
            "persona:document.querySelectorAll('.memory-detail-content').length>=2,"
            "source:Boolean(document.querySelector('#memory-detail-source')),"
            "archive:Boolean(document.querySelector('#memory-detail-archive')),"
            "rawMetadataCollapsed:Boolean(document.querySelector("
            "'.memory-detail-section:not([open]) > summary'))"
            "})"
        )
        await screenshot("livingmemory-253-detail-dark-desktop.png")
        await evaluate("document.querySelector('#memory-detail-source').click()")
        await wait_for("document.querySelectorAll('.memory-source-message').length>=2")
        source_detail = await evaluate(
            "({"
            "messages:document.querySelectorAll('.memory-source-message').length,"
            "overflow:document.querySelector('#memory-detail-panel').scrollWidth>"
            "document.querySelector('#memory-detail-panel').clientWidth"
            "})"
        )
        await screenshot("livingmemory-253-source-dark-desktop.png")
        await evaluate("document.querySelector('#memory-detail-close').click()")

        await evaluate(
            "(()=>{const select=document.querySelector('#memory-status');"
            "select.value='archived';"
            "select.dispatchEvent(new Event('change',{bubbles:true}));return true})()"
        )
        await wait_for("document.querySelectorAll('#memory-rows .memory-row').length===1")
        await evaluate("document.querySelector('#memory-rows .memory-row').click()")
        await wait_for("Boolean(document.querySelector('#memory-detail-restore'))")
        await wait_for(
            "document.querySelector('#memory-detail-panel').getBoundingClientRect().right"
            "<=window.innerWidth+1"
        )
        archived_detail = await evaluate(
            "({"
            "restore:Boolean(document.querySelector('#memory-detail-restore')),"
            "source:Boolean(document.querySelector('#memory-detail-source'))"
            "})"
        )
        await screenshot("livingmemory-253-archived-detail-dark-desktop.png")
        await evaluate("document.querySelector('#memory-detail-close').click()")

        await open_database_page("graph")
        await evaluate("document.querySelector('#graph-overview').click()")
        await wait_for("Boolean(document.querySelector('#graph-canvas canvas'))")
        await asyncio.sleep(2.1)
        graph = await evaluate(
            "({"
            "canvas:Boolean(document.querySelector('#graph-canvas canvas')),"
            "legend:document.querySelectorAll('#graph-legend span').length,"
            "bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth,"
            "canvasWidth:Math.round(document.querySelector('#graph-canvas canvas')"
            ".getBoundingClientRect().width)"
            "})"
        )
        await screenshot("livingmemory-253-graph-dark-desktop.png")
        await set_theme("light")
        await screenshot("livingmemory-253-graph-light-desktop.png")

        await evaluate("document.querySelector('[data-page=\"libraries\"]').click()")
        await wait_for(
            "[...document.querySelectorAll('.library-card')].some("
            f"item=>item.dataset.id==={json.dumps(database_id)})"
        )
        edited = await evaluate(
            "(()=>{const card=[...document.querySelectorAll('.library-card')].find("
            f"item=>item.dataset.id==={json.dumps(database_id)}"
            ");const button=card?.querySelector('.edit-library');"
            "if(!button)return false;button.click();return true})()"
        )
        if not edited:
            raise RuntimeError("LivingMemory settings editor could not be opened")
        await wait_for(
            "!document.querySelector('#library-modal').classList.contains('hidden')"
            "&&Boolean(document.querySelector('#library-min-importance-retrieval'))"
        )
        settings = await evaluate(
            "({"
            "minImportance:Boolean(document.querySelector('#library-min-importance-retrieval')),"
            "minSimilarity:Boolean(document.querySelector('#library-min-similarity-retrieval')),"
            "recentCount:Boolean(document.querySelector('#library-recent-memory-count')),"
            "recentAge:Boolean(document.querySelector('#library-recent-memory-max-age')),"
            "typeFilter:Boolean(document.querySelector('#library-memory-type-filter')),"
            "autoArchive:Boolean(document.querySelector('#library-auto-archived-enabled')),"
            "protectedImportance:Boolean(document.querySelector('#library-protected-importance-threshold')),"
            "overflow:document.querySelector('#library-form').scrollWidth>"
            "document.querySelector('#library-form').clientWidth"
            "})"
        )
        await screenshot("livingmemory-253-settings-light-desktop.png")
        await evaluate(
            "document.querySelector('#library-protected-importance-threshold')"
            ".scrollIntoView({block:'center'})"
        )
        await set_theme("dark")
        await screenshot("livingmemory-253-settings-maintenance-dark-desktop.png")

        await evaluate("document.querySelector('#library-modal .modal-dismiss').click()")
        await call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 390,
                "height": 844,
                "deviceScaleFactor": 1,
                "mobile": True,
            },
        )
        await evaluate(
            "(()=>{const card=[...document.querySelectorAll('.library-card')].find("
            f"item=>item.dataset.id==={json.dumps(database_id)}"
            ");card?.querySelector('.enter-library')?.click();return Boolean(card)})()"
        )
        await wait_for(
            "Boolean(document.querySelector("
            "'[data-route=\"database:livingmemory_v8:memories\"]'))"
        )
        await open_database_page("memories")
        await evaluate(
            "(()=>{const select=document.querySelector('#memory-status');"
            "select.value='';"
            "select.dispatchEvent(new Event('change',{bubbles:true}));return true})()"
        )
        await wait_for("document.querySelectorAll('#memory-rows .memory-row').length>0")
        mobile = await evaluate(
            "({"
            "viewport:window.innerWidth,"
            "bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth,"
            "pageOverflow:document.querySelector('#page-memory').scrollWidth>"
            "document.querySelector('#page-memory').clientWidth,"
            "transferWidth:Math.round(document.querySelector('.memory-transfer-panel')"
            ".getBoundingClientRect().width)"
            "})"
        )
        await screenshot("livingmemory-253-memories-dark-mobile.png")
        await evaluate(
            "document.querySelector('.memory-transfer-panel')"
            ".scrollIntoView({block:'start'})"
        )
        await asyncio.sleep(0.2)
        await screenshot("livingmemory-253-transfer-dark-mobile.png")
        await evaluate("document.querySelector('#memory-rows .memory-row').click()")
        await wait_for(
            "document.querySelector('#memory-detail-panel')?.classList.contains('visible')"
            "&&document.querySelector('#memory-detail-panel').getBoundingClientRect().right"
            "<=window.innerWidth+1"
        )
        mobile_detail = await evaluate(
            "({"
            "width:Math.round(document.querySelector('#memory-detail-panel')"
            ".getBoundingClientRect().width),"
            "overflow:document.querySelector('#memory-detail-panel').scrollWidth>"
            "document.querySelector('#memory-detail-panel').clientWidth"
            "})"
        )
        await screenshot("livingmemory-253-detail-dark-mobile.png")
        await evaluate("document.querySelector('#memory-detail-close').click()")
        await open_database_page("graph")
        await wait_for("Boolean(document.querySelector('.graph-toolbar'))")
        await evaluate(
            "document.querySelector('.graph-toolbar')"
            ".scrollIntoView({block:'start'})"
        )
        mobile_graph = await evaluate(
            "({"
            "pageOverflow:document.querySelector('#page-graph').scrollWidth>"
            "document.querySelector('#page-graph').clientWidth,"
            "toolbarOverflow:document.querySelector('.graph-toolbar').scrollWidth>"
            "document.querySelector('.graph-toolbar').clientWidth,"
            "toolbarWidth:Math.round(document.querySelector('.graph-toolbar')"
            ".getBoundingClientRect().width),"
            "widestControl:Math.max(...[...document.querySelectorAll("
            "'.graph-toolbar > *')].map(item=>item.getBoundingClientRect().width))"
            "})"
        )
        await screenshot("livingmemory-253-graph-dark-mobile.png")

        return {
            "memory_desktop": memory_desktop,
            "detail": detail,
            "source_detail": source_detail,
            "archived_detail": archived_detail,
            "graph": graph,
            "settings": settings,
            "mobile": mobile,
            "mobile_detail": mobile_detail,
            "mobile_graph": mobile_graph,
            "browser_exceptions": len(exceptions),
        }


def acceptance_failures(report: dict[str, object]) -> list[str]:
    failures: list[str] = []

    def require(condition: bool, label: str) -> None:
        if not condition:
            failures.append(label)

    memory = report["memory_desktop"]
    require(memory["rows"] > 0, "desktop memory rows are empty")
    require(memory["transfer"], "portable transfer panel is missing")
    require(not memory["bodyOverflow"], "desktop body has horizontal overflow")
    require(not memory["pageOverflow"], "desktop memory page has horizontal overflow")

    detail = report["detail"]
    for field in ("canonical", "persona", "source", "archive", "rawMetadataCollapsed"):
        require(detail[field], f"active memory detail check failed: {field}")

    source = report["source_detail"]
    require(source["messages"] >= 2, "source messages were not rendered")
    require(not source["overflow"], "source detail has horizontal overflow")

    archived = report["archived_detail"]
    require(archived["restore"], "archived memory restore action is missing")
    require(archived["source"], "archived memory source action is missing")

    graph = report["graph"]
    require(graph["canvas"], "graph canvas is missing")
    require(graph["legend"] > 0, "graph legend is empty")
    require(not graph["bodyOverflow"], "graph page has horizontal overflow")
    require(graph["canvasWidth"] > 0, "graph canvas has no visible width")

    settings = report["settings"]
    for field in (
        "minImportance",
        "minSimilarity",
        "recentCount",
        "recentAge",
        "typeFilter",
        "autoArchive",
        "protectedImportance",
    ):
        require(settings[field], f"LivingMemory 2.5.3 setting is missing: {field}")
    require(not settings["overflow"], "library settings form has horizontal overflow")

    mobile = report["mobile"]
    require(mobile["viewport"] == 390, "mobile viewport override failed")
    require(not mobile["bodyOverflow"], "mobile body has horizontal overflow")
    require(not mobile["pageOverflow"], "mobile memory page has horizontal overflow")
    require(0 < mobile["transferWidth"] <= 390, "mobile transfer panel width is invalid")

    mobile_detail = report["mobile_detail"]
    require(0 < mobile_detail["width"] <= 390, "mobile memory detail width is invalid")
    require(not mobile_detail["overflow"], "mobile memory detail has horizontal overflow")
    mobile_graph = report["mobile_graph"]
    require(not mobile_graph["pageOverflow"], "mobile graph page has horizontal overflow")
    require(not mobile_graph["toolbarOverflow"], "mobile graph toolbar has horizontal overflow")
    require(
        0 < mobile_graph["widestControl"] <= mobile_graph["toolbarWidth"] + 1,
        "mobile graph toolbar controls exceed the panel width",
    )
    require(report["browser_exceptions"] == 0, "browser runtime exceptions were captured")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devtools-url", default="http://127.0.0.1:9227")
    parser.add_argument("--base-url", default="http://127.0.0.1:8877")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--database-id", default="lm253_acceptance")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(
        run(
            args.devtools_url,
            base_url=args.base_url,
            api_key=args.api_key,
            database_id=args.database_id,
            report_dir=args.report_dir,
        )
    )
    report["acceptance_failures"] = acceptance_failures(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["acceptance_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

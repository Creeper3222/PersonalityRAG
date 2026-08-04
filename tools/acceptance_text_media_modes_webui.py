from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import urllib.request

import websockets


async def run(
    devtools_url: str,
    base_url: str,
    api_key: str,
    report_dir: Path,
) -> dict:
    pages = json.loads(
        urllib.request.urlopen(f"{devtools_url}/json/list", timeout=5).read()
    )
    page = next(item for item in pages if item.get("type") == "page")
    exceptions: list[dict] = []
    async with websockets.connect(
        page["webSocketDebuggerUrl"], max_size=20_000_000
    ) as ws:
        counter = 0

        async def call(method: str, params: dict | None = None) -> dict:
            nonlocal counter
            counter += 1
            request_id = counter
            await ws.send(
                json.dumps(
                    {"id": request_id, "method": method, "params": params or {}}
                )
            )
            while True:
                payload = json.loads(await ws.recv())
                if payload.get("method") == "Runtime.exceptionThrown":
                    exceptions.append(payload)
                if payload.get("id") == request_id:
                    if payload.get("error"):
                        raise RuntimeError(payload["error"])
                    return payload.get("result", {})

        async def evaluate(expression: str):
            response = await call(
                "Runtime.evaluate",
                {
                    "expression": expression,
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            )
            return response.get("result", {}).get("value")

        async def wait_for(expression: str, timeout: float = 20.0) -> None:
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                if await evaluate(f"Boolean({expression})"):
                    return
                await asyncio.sleep(0.2)
            raise TimeoutError(expression)

        async def screenshot(name: str) -> None:
            response = await call(
                "Page.captureScreenshot",
                {"format": "png", "captureBeyondViewport": False},
            )
            (report_dir / name).write_bytes(base64.b64decode(response["data"]))

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
        await wait_for("document.querySelector('#library-cards .library-card')")
        await evaluate(
            "document.querySelector('.nav[data-page=libraries]')?.click();"
            "document.querySelector('.database-category-tab[data-database-category=knowledge]')?.click()"
        )
        await wait_for(
            "document.querySelector('.library-card[data-id=beileite_test] .manage-text-media')"
        )
        await evaluate(
            "document.querySelector('.library-card[data-id=beileite_test] .edit-text-media').click()"
        )
        await wait_for(
            "!document.querySelector('#text-media-edit-modal').classList.contains('hidden')"
        )
        await evaluate(
            "document.querySelector('#text-media-edit-lexical-common-floor').scrollIntoView({block:'center'})"
        )
        await asyncio.sleep(0.2)
        edit_settings = await evaluate(
            """({
              commonFloor:Number(document.querySelector('#text-media-edit-lexical-common-floor').value),
              oovPenalty:Number(document.querySelector('#text-media-edit-lexical-oov-penalty').value),
              rarityExponent:Number(document.querySelector('#text-media-edit-distinctive-rarity-exponent').value),
              unboundCandidateLimit:Number(document.querySelector('#text-media-edit-unbound-candidate-limit').value),
              unboundDistinctiveBoost:Number(document.querySelector('#text-media-edit-unbound-distinctive-boost').value),
              unboundCollectionBoost:Number(document.querySelector('#text-media-edit-unbound-collection-boost').value),
              unboundCompetitionFloor:Number(document.querySelector('#text-media-edit-unbound-competition-floor').value),
              unboundReliabilityTarget:Number(document.querySelector('#text-media-edit-unbound-reliability-target').value),
              unboundSpecificityExponent:Number(document.querySelector('#text-media-edit-unbound-specificity-exponent').value),
              unboundAdvantageTarget:Number(document.querySelector('#text-media-edit-unbound-advantage-target').value),
              boundDistinctiveBoost:Number(document.querySelector('#text-media-edit-bound-distinctive-boost').value),
              boundRescueMin:Number(document.querySelector('#text-media-edit-bound-distinctive-rescue-min').value),
              labels:['常见描述词最低贡献','语料外查询词权重','显著特征词稀有度指数','无绑定媒体候选窗口','无绑定媒体显著特征词加成','无绑定媒体集合词加成','无绑定媒体软竞争底座','无绑定媒体可靠性目标','无绑定媒体特异度指数','无绑定媒体优势目标','绑定媒体显著特征词加成','绑定媒体显著候选补充下限'].every(label=>document.querySelector('#text-media-edit-modal').textContent.includes(label)),
              overflow:document.querySelector('#text-media-edit-modal .modal').scrollWidth > document.querySelector('#text-media-edit-modal .modal').clientWidth + 1
            })"""
        )
        await screenshot("text-media-distinctive-edit-desktop.png")
        await call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 430,
                "height": 900,
                "deviceScaleFactor": 1,
                "mobile": True,
            },
        )
        await asyncio.sleep(0.3)
        await evaluate(
            "document.querySelector('#text-media-edit-unbound-competition-floor').scrollIntoView({block:'center'})"
        )
        await asyncio.sleep(0.2)
        edit_settings_mobile = await evaluate(
            """({
              bodyOverflow:document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
              modalOverflow:document.querySelector('#text-media-edit-modal .modal').scrollWidth > document.querySelector('#text-media-edit-modal .modal').clientWidth + 1,
              settingsColumns:getComputedStyle(document.querySelector('#text-media-edit-modal .text-media-retrieval-settings-grid')).gridTemplateColumns
            })"""
        )
        await screenshot("text-media-distinctive-edit-mobile.png")
        await evaluate(
            "document.querySelector('#text-media-edit-modal .modal-dismiss').click()"
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
        await evaluate(
            "document.querySelector('.library-card[data-id=beileite_test] .manage-text-media').click()"
        )
        await wait_for("document.querySelector('.text-media-workspace-page')")

        await evaluate("document.querySelector('#text-media-ingest-open').click()")
        await wait_for(
            "!document.querySelector('#text-media-ingest-modal').classList.contains('hidden')"
        )
        await evaluate(
            """(() => {
              const transfer = new DataTransfer();
              transfer.items.add(new File(['pure text body'], 'pure-text.txt', {type:'text/plain'}));
              const input = document.querySelector('#text-media-ingest-documents');
              input.files = transfer.files;
              input.dispatchEvent(new Event('change', {bubbles:true}));
            })()"""
        )
        await wait_for(
            "document.querySelectorAll('.text-media-ingest-file-row').length===1"
        )
        pure_text = await evaluate(
            """({
              mode:document.querySelector('#text-media-ingest-mode').textContent,
              documents:document.querySelectorAll('.text-media-ingest-file-row').length,
              images:document.querySelectorAll('.text-media-ingest-image-row').length,
              chunkTargetDisabled:document.querySelector('#text-media-ingest-chunk-target').disabled,
              chunkOverlapDisabled:document.querySelector('#text-media-ingest-chunk-overlap').disabled,
              semanticHidden:document.querySelector('#text-media-ingest-semantic-section').classList.contains('hidden'),
              overflow:document.querySelector('#text-media-ingest-modal .modal').scrollWidth > document.querySelector('#text-media-ingest-modal .modal').clientWidth + 1
            })"""
        )
        await screenshot("text-media-pure-text-ingest-desktop.png")
        await evaluate(
            "document.querySelector('#text-media-ingest-modal .modal-dismiss').click();"
            "document.querySelector('#text-media-ingest-open').click()"
        )
        await wait_for(
            "!document.querySelector('#text-media-ingest-modal').classList.contains('hidden')"
        )
        await evaluate(
            """(() => {
              const transfer = new DataTransfer();
              transfer.items.add(new File([new Uint8Array([255,216,255,217])], '原来是劣等模型 Claude版表情包.jpg', {type:'image/jpeg'}));
              const input = document.querySelector('#text-media-ingest-images');
              input.files = transfer.files;
              input.dispatchEvent(new Event('change', {bubbles:true}));
            })()"""
        )
        await wait_for(
            "document.querySelectorAll('.text-media-ingest-image-row').length===1"
        )
        pure_media = await evaluate(
            """({
              mode:document.querySelector('#text-media-ingest-mode').textContent,
              documents:document.querySelectorAll('.text-media-ingest-file-row').length,
              images:document.querySelectorAll('.text-media-ingest-image-row').length,
              description:document.querySelector('.text-media-ingest-description').value,
              noBinding:document.querySelector('.text-media-ingest-scope input[value=none]').checked,
              chunkTargetDisabled:document.querySelector('#text-media-ingest-chunk-target').disabled,
              chunkOverlapDisabled:document.querySelector('#text-media-ingest-chunk-overlap').disabled,
              semanticHidden:document.querySelector('#text-media-ingest-semantic-section').classList.contains('hidden'),
              overflow:document.querySelector('#text-media-ingest-modal .modal').scrollWidth > document.querySelector('#text-media-ingest-modal .modal').clientWidth + 1
            })"""
        )
        await screenshot("text-media-pure-media-ingest-desktop.png")
        await call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 430,
                "height": 900,
                "deviceScaleFactor": 1,
                "mobile": True,
            },
        )
        await asyncio.sleep(0.3)
        pure_media_mobile = await evaluate(
            """({
              bodyOverflow:document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
              modalOverflow:document.querySelector('#text-media-ingest-modal .modal').scrollWidth > document.querySelector('#text-media-ingest-modal .modal').clientWidth + 1,
              pickerColumns:getComputedStyle(document.querySelector('.text-media-ingest-pickers')).gridTemplateColumns
            })"""
        )
        await screenshot("text-media-pure-media-ingest-mobile.png")
        await evaluate(
            "document.querySelector('#text-media-ingest-modal .modal-dismiss').click()"
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
        await evaluate(
            "document.querySelector('[data-route=\"database:text_media_v1:search\"]').click();"
            "document.querySelector('#text-media-search-mode-standard').click();"
            "document.querySelector('#text-media-search-query').value='给我deepseek表情包';"
            "document.querySelector('#text-media-search-form').requestSubmit()"
        )
        await wait_for(
            "document.querySelectorAll('#text-media-search-media .text-media-search-media-card').length===1",
            timeout=30,
        )
        standard = await evaluate(
            """({
              standardActive:document.querySelector('#text-media-search-mode-standard').classList.contains('active'),
              textOnlyActive:document.querySelector('#text-media-search-mode-text-only').classList.contains('active'),
              mediaOnlyActive:document.querySelector('#text-media-search-mode-media-only').classList.contains('active'),
              topKDisabled:document.querySelector('#text-media-search-top-k').disabled,
              chunksHidden:document.querySelector('#text-media-search-chunks-section').classList.contains('hidden'),
              mediaHidden:document.querySelector('#text-media-search-media-section').classList.contains('hidden'),
              decisionsHidden:document.querySelector('#text-media-search-decisions-section').classList.contains('hidden'),
              outputs:document.querySelectorAll('#text-media-search-media .text-media-search-media-card').length,
              outputText:document.querySelector('#text-media-search-media').innerText
            })"""
        )
        await evaluate(
            "document.querySelector('#text-media-search-mode-text-only').click();"
            "document.querySelector('#text-media-search-query').value='澄月';"
            "document.querySelector('#text-media-search-form').requestSubmit()"
        )
        await wait_for(
            "document.querySelectorAll('#text-media-search-results .text-media-result').length>0",
            timeout=30,
        )
        text_only = await evaluate(
            """({
              standardActive:document.querySelector('#text-media-search-mode-standard').classList.contains('active'),
              textOnlyActive:document.querySelector('#text-media-search-mode-text-only').classList.contains('active'),
              mediaOnlyActive:document.querySelector('#text-media-search-mode-media-only').classList.contains('active'),
              topKDisabled:document.querySelector('#text-media-search-top-k').disabled,
              chunksHidden:document.querySelector('#text-media-search-chunks-section').classList.contains('hidden'),
              mediaHidden:document.querySelector('#text-media-search-media-section').classList.contains('hidden'),
              decisionsHidden:document.querySelector('#text-media-search-decisions-section').classList.contains('hidden'),
              results:document.querySelectorAll('#text-media-search-results .text-media-result').length,
              outputs:document.querySelectorAll('#text-media-search-media .text-media-search-media-card').length,
              decisions:document.querySelectorAll('#text-media-search-decisions .text-media-decision').length
            })"""
        )
        await evaluate(
            "document.querySelector('#text-media-search-mode-media-only').click();"
            "document.querySelector('#text-media-search-query').value='给我deepseek表情包';"
            "document.querySelector('#text-media-search-form').requestSubmit()"
        )
        await wait_for(
            "document.querySelectorAll('#text-media-search-media .text-media-search-media-card').length===1",
            timeout=30,
        )
        media_only = await evaluate(
            """({
              standardActive:document.querySelector('#text-media-search-mode-standard').classList.contains('active'),
              mediaOnlyActive:document.querySelector('#text-media-search-mode-media-only').classList.contains('active'),
              topKDisabled:document.querySelector('#text-media-search-top-k').disabled,
              chunksHidden:document.querySelector('#text-media-search-chunks-section').classList.contains('hidden'),
              outputs:document.querySelectorAll('#text-media-search-media .text-media-search-media-card').length,
              outputText:document.querySelector('#text-media-search-media').innerText,
              decisions:document.querySelectorAll('#text-media-search-decisions .text-media-decision').length,
              thresholdNotApplicable:document.querySelector('#text-media-search-decisions').innerText.includes('不适用'),
              competitionLabels:['无绑定媒体竞争前置信度','无绑定媒体简述直接置信度','无绑定媒体直接描述可靠性','无绑定媒体相对目标特异度','无绑定媒体单目标赢家支持','无绑定媒体软竞争底座','无绑定媒体置信竞争系数','无绑定媒体集合请求'].every(label=>document.querySelector('#text-media-search-decisions').textContent.includes(label)),
              frequencyLabels:['媒体词频作用域','频率加权完整度','命中词信息量','显著特征支持','显著集合成员支持','集合类别支持','集合成员综合支持','最强无绑定竞争媒体置信度','无绑定媒体目标竞争优势','无绑定媒体单目标优势权重','无绑定媒体集合成员权重'].every(label=>document.querySelector('#text-media-search-decisions').textContent.includes(label)),
              allDetailsCollapsed:[...document.querySelectorAll('#text-media-search-decisions details')].every(item=>!item.open),
              overflow:document.querySelector('.text-media-workspace-page').scrollWidth > document.querySelector('.text-media-workspace-page').clientWidth + 1
            })"""
        )
        await screenshot("text-media-media-only-search-desktop.png")
        await evaluate(
            "document.querySelector('#text-media-search-decisions .text-media-diagnostic-details').open=true;"
            "document.querySelector('#text-media-search-decisions .text-media-diagnostic-details').scrollIntoView({block:'center'})"
        )
        await asyncio.sleep(0.2)
        await screenshot("text-media-media-only-details-desktop.png")
        await call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 430,
                "height": 900,
                "deviceScaleFactor": 1,
                "mobile": True,
            },
        )
        await asyncio.sleep(0.3)
        media_only_mobile = await evaluate(
            """({
              bodyOverflow:document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
              workspaceOverflow:document.querySelector('.text-media-workspace-page').scrollWidth > document.querySelector('.text-media-workspace-page').clientWidth + 1,
              formColumns:getComputedStyle(document.querySelector('#text-media-search-form')).gridTemplateColumns
            })"""
        )
        await screenshot("text-media-media-only-search-mobile.png")
        await evaluate(
            "document.querySelector('#text-media-search-top-k').value='3';"
            "document.querySelector('#text-media-search-confidence-threshold').value='0.2';"
            "document.querySelector('#text-media-search-score-threshold').value='0.8';"
            "document.querySelector('#text-media-search-max-outputs').value='1';"
            "document.querySelector('#text-media-page-refresh').click()"
        )
        await asyncio.sleep(0.3)
        refresh = await evaluate(
            """({
              query:document.querySelector('#text-media-search-query').value,
              standardActive:document.querySelector('#text-media-search-mode-standard').classList.contains('active'),
              textOnlyActive:document.querySelector('#text-media-search-mode-text-only').classList.contains('active'),
              mediaOnlyActive:document.querySelector('#text-media-search-mode-media-only').classList.contains('active'),
              topK:Number(document.querySelector('#text-media-search-top-k').value),
              topKDisabled:document.querySelector('#text-media-search-top-k').disabled,
              confidence:Number(document.querySelector('#text-media-search-confidence-threshold').value),
              score:Number(document.querySelector('#text-media-search-score-threshold').value),
              maxOutputs:Number(document.querySelector('#text-media-search-max-outputs').value),
              chunksHidden:document.querySelector('#text-media-search-chunks-section').classList.contains('hidden'),
              outputs:document.querySelector('#text-media-search-media').childElementCount,
              decisions:document.querySelector('#text-media-search-decisions').childElementCount,
              results:document.querySelector('#text-media-search-results').childElementCount
            })"""
        )
    return {
        "edit_settings": edit_settings,
        "edit_settings_mobile": edit_settings_mobile,
        "pure_text": pure_text,
        "pure_media": pure_media,
        "pure_media_mobile": pure_media_mobile,
        "standard": standard,
        "text_only": text_only,
        "media_only": media_only,
        "media_only_mobile": media_only_mobile,
        "refresh": refresh,
        "runtime_exceptions": len(exceptions),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devtools-url", default="http://127.0.0.1:9225")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--api-key",
        default=os.getenv("PERSONALITYRAG_API_KEY", ""),
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.api_key:
        parser.error(
            "--api-key or the PERSONALITYRAG_API_KEY environment variable is required"
        )
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(
        run(args.devtools_url, args.base_url, args.api_key, report_dir)
    )
    (report_dir / "acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    passed = (
        result["runtime_exceptions"] == 0
        and result["edit_settings"]
        == {
            "commonFloor": 0.0,
            "oovPenalty": 0.3,
            "rarityExponent": 1.5,
            "unboundCandidateLimit": 10,
            "unboundDistinctiveBoost": 0.35,
            "unboundCollectionBoost": 0.55,
            "unboundCompetitionFloor": 0.35,
            "unboundReliabilityTarget": 0.25,
            "unboundSpecificityExponent": 1.0,
            "unboundAdvantageTarget": 0.04,
            "boundDistinctiveBoost": 0.1,
            "boundRescueMin": 0.8,
            "labels": True,
            "overflow": False,
        }
        and not result["edit_settings_mobile"]["bodyOverflow"]
        and not result["edit_settings_mobile"]["modalOverflow"]
        and result["pure_text"]["documents"] == 1
        and result["pure_text"]["images"] == 0
        and not result["pure_text"]["chunkTargetDisabled"]
        and not result["pure_text"]["chunkOverlapDisabled"]
        and result["pure_text"]["semanticHidden"]
        and not result["pure_text"]["overflow"]
        and result["pure_media"]["documents"] == 0
        and result["pure_media"]["images"] == 1
        and result["pure_media"]["description"]
        == "原来是劣等模型 Claude版表情包"
        and result["pure_media"]["noBinding"]
        and result["pure_media"]["chunkTargetDisabled"]
        and result["pure_media"]["chunkOverlapDisabled"]
        and result["pure_media"]["semanticHidden"]
        and not result["pure_media"]["overflow"]
        and not result["pure_media_mobile"]["bodyOverflow"]
        and not result["pure_media_mobile"]["modalOverflow"]
        and result["standard"]["standardActive"]
        and not result["standard"]["textOnlyActive"]
        and not result["standard"]["mediaOnlyActive"]
        and not result["standard"]["topKDisabled"]
        and not result["standard"]["chunksHidden"]
        and not result["standard"]["mediaHidden"]
        and not result["standard"]["decisionsHidden"]
        and result["standard"]["outputs"] == 1
        and "deepseek" in result["standard"]["outputText"].casefold()
        and not result["text_only"]["standardActive"]
        and result["text_only"]["textOnlyActive"]
        and not result["text_only"]["mediaOnlyActive"]
        and not result["text_only"]["topKDisabled"]
        and not result["text_only"]["chunksHidden"]
        and result["text_only"]["mediaHidden"]
        and result["text_only"]["decisionsHidden"]
        and result["text_only"]["results"] > 0
        and result["text_only"]["outputs"] == 0
        and result["text_only"]["decisions"] == 0
        and not result["media_only"]["standardActive"]
        and result["media_only"]["mediaOnlyActive"]
        and result["media_only"]["topKDisabled"]
        and result["media_only"]["chunksHidden"]
        and result["media_only"]["outputs"] == 1
        and "deepseek" in result["media_only"]["outputText"].casefold()
        and result["media_only"]["decisions"] == 8
        and result["media_only"]["thresholdNotApplicable"]
        and result["media_only"]["competitionLabels"]
        and result["media_only"]["frequencyLabels"]
        and result["media_only"]["allDetailsCollapsed"]
        and not result["media_only"]["overflow"]
        and not result["media_only_mobile"]["bodyOverflow"]
        and not result["media_only_mobile"]["workspaceOverflow"]
        and result["refresh"]
        == {
            "query": "给我deepseek表情包",
            "standardActive": True,
            "textOnlyActive": False,
            "mediaOnlyActive": False,
            "topK": 10,
            "topKDisabled": False,
            "confidence": 0.6,
            "score": 0.35,
            "maxOutputs": 5,
            "chunksHidden": False,
            "outputs": 0,
            "decisions": 0,
            "results": 0,
        }
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import urllib.request
from pathlib import Path

import websockets


async def run(
    devtools_url: str, base_url: str, api_key: str, report_dir: Path
) -> dict:
    pages = json.loads(
        urllib.request.urlopen(f"{devtools_url}/json/list", timeout=5).read()
    )
    page = next(item for item in pages if item.get("type") == "page")
    errors: list[dict] = []
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
                    errors.append(payload)
                if payload.get("id") == request_id:
                    if payload.get("error"):
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

        async def wait_for(expression: str, timeout: float = 12.0) -> None:
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                if await evaluate(f"Boolean({expression})"):
                    return
                await asyncio.sleep(0.2)
            raise TimeoutError(expression)

        async def screenshot(name: str) -> None:
            result = await call(
                "Page.captureScreenshot",
                {"format": "png", "captureBeyondViewport": False},
            )
            (report_dir / name).write_bytes(base64.b64decode(result["data"]))

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
        await call("Page.navigate", {"url": f"{base_url.rstrip('/')}/"})
        await wait_for("document.querySelector('#library-cards .library-card')")
        await evaluate(
            "document.querySelector('.nav[data-page=libraries]')?.click();"
            "document.querySelector('.database-category-tab[data-database-category=knowledge]')?.click()"
        )
        await wait_for(
            "document.querySelector('.library-card[data-id=beileite_test] .edit-text-media')"
        )
        library_card = await evaluate(
            "({text:document.querySelector('.library-card[data-id=beileite_test]').innerText,"
            "rerank:[...document.querySelectorAll('.library-card[data-id=beileite_test] .library-card-extra dt')]"
            ".some(item=>item.textContent.includes('重排')&&item.nextElementSibling?.textContent.trim()&&item.nextElementSibling.textContent.trim()!=='无')})"
        )
        await evaluate(
            "document.querySelector('.library-card[data-id=beileite_test] .edit-text-media').click()"
        )
        await wait_for(
            "!document.querySelector('#text-media-edit-modal').classList.contains('hidden')"
        )
        await evaluate(
            "document.querySelector('#text-media-retrieval-settings').scrollIntoView({block:'start'})"
        )
        await asyncio.sleep(0.5)
        desktop_edit = await evaluate(
            """({
              settings:document.querySelectorAll('#text-media-edit-modal .text-media-retrieval-settings input').length,
              rerankProvider:document.querySelector('#text-media-edit-rerank-provider').value,
              rerankCandidate:Number(document.querySelector('#text-media-edit-rerank-candidate-limit').value),
              rerankFusionWeight:Number(document.querySelector('#text-media-edit-rerank-fusion-weight').value),
              rerankRankBonus:Number(document.querySelector('#text-media-edit-rerank-rank-bonus-weight').value),
              rerankReliabilityExponent:Number(document.querySelector('#text-media-edit-rerank-reliability-exponent').value),
              textLexicalBoost:Number(document.querySelector('#text-media-edit-text-lexical-boost').value),
              rerankFusionWeightValid:document.querySelector('#text-media-edit-rerank-fusion-weight').checkValidity(),
              candidate:Number(document.querySelector('#text-media-edit-candidate-limit').value),
              floor:Number(document.querySelector('#text-media-edit-semantic-floor').value),
              semanticWeight:Number(document.querySelector('#text-media-edit-semantic-weight').value),
              lexicalBoost:Number(document.querySelector('#text-media-edit-lexical-boost').value),
              thresholdFallback:Number(document.querySelector('#text-media-edit-score-threshold-fallback').value),
              thresholdEvidenceLimit:Number(document.querySelector('#text-media-edit-threshold-evidence-limit').value),
              thresholdRankExponent:Number(document.querySelector('#text-media-edit-threshold-rank-exponent').value),
              thresholdNegativeExponent:Number(document.querySelector('#text-media-edit-threshold-negative-exponent').value),
              thresholdReinforcement:Number(document.querySelector('#text-media-edit-threshold-reinforcement-weight').value),
              thresholdWeakening:Number(document.querySelector('#text-media-edit-threshold-weakening-weight').value),
              pivotNegativeFloor:Number(document.querySelector('#text-media-edit-pivot-negative-floor').value),
              formatMismatchFactor:Number(document.querySelector('#text-media-edit-format-mismatch-factor').value),
              contentMismatchFactor:Number(document.querySelector('#text-media-edit-content-mismatch-factor').value),
              gate:document.querySelector('#text-media-edit-intent-gate').checked,
              gateRole:document.querySelector('#text-media-edit-intent-gate').getAttribute('role'),
              gateSwitchWidth:Math.round(document.querySelector('.text-media-intent-gate-switch').getBoundingClientRect().width),
              moduleGap:Math.round(document.querySelector('.text-media-visual-intent-policy').getBoundingClientRect().top-document.querySelector('#text-media-retrieval-settings').getBoundingClientRect().bottom),
              overflow:document.querySelector('#text-media-edit-modal .modal').scrollWidth > document.querySelector('#text-media-edit-modal .modal').clientWidth + 1
            })"""
        )
        await screenshot("text-media-edit-desktop.png")
        desktop_retrieval_compact = await evaluate(
            """(()=>{const section=document.querySelector('#text-media-retrieval-settings');const primary=document.querySelector('#text-media-retrieval-primary');const toggle=document.querySelector('#text-media-retrieval-toggle');return {
              layout:section.dataset.layout,
              columns:getComputedStyle(document.querySelector('.text-media-retrieval-settings-grid')).gridTemplateColumns.split(' ').length,
              visibleHints:[...section.querySelectorAll('small')].filter(item=>item.getClientRects().length).length,
              collapsed:section.classList.contains('text-media-retrieval-collapsed'),
              toggleVisible:!toggle.classList.contains('hidden'),
              aria:toggle.getAttribute('aria-expanded'),
              clientHeight:primary.clientHeight,
              expectedHeight:Math.round(window.innerHeight*0.75),
              scrollHeight:primary.scrollHeight
            };})()"""
        )
        await evaluate("document.querySelector('#text-media-retrieval-toggle').click()")
        await asyncio.sleep(0.5)
        desktop_retrieval_expanded = await evaluate(
            """(()=>{const section=document.querySelector('#text-media-retrieval-settings');const primary=document.querySelector('#text-media-retrieval-primary');const toggle=document.querySelector('#text-media-retrieval-toggle');return {expanded:section.classList.contains('text-media-retrieval-expanded'),aria:toggle.getAttribute('aria-expanded'),fullyVisible:primary.clientHeight===primary.scrollHeight};})()"""
        )
        await screenshot("text-media-retrieval-compact-expanded-desktop.png")
        await evaluate("document.querySelector('[data-retrieval-layout=detailed]').click()")
        await asyncio.sleep(0.5)
        desktop_retrieval_detailed = await evaluate(
            """(()=>{const section=document.querySelector('#text-media-retrieval-settings');const primary=document.querySelector('#text-media-retrieval-primary');return {layout:section.dataset.layout,columns:getComputedStyle(section.querySelector('.text-media-retrieval-settings-grid')).gridTemplateColumns.split(' ').length,visibleHints:[...section.querySelectorAll('.text-media-retrieval-settings-grid small')].filter(item=>item.getClientRects().length).length,expanded:section.classList.contains('text-media-retrieval-expanded'),fullyVisible:primary.clientHeight===primary.scrollHeight};})()"""
        )
        await screenshot("text-media-retrieval-detailed-expanded-desktop.png")
        await evaluate("document.querySelector('#text-media-retrieval-toggle').click()")
        await asyncio.sleep(0.5)
        desktop_retrieval_detailed_collapsed = await evaluate(
            """(()=>{const section=document.querySelector('#text-media-retrieval-settings');const primary=document.querySelector('#text-media-retrieval-primary');return {collapsed:section.classList.contains('text-media-retrieval-collapsed'),clientHeight:primary.clientHeight,expectedHeight:Math.round(window.innerHeight*0.75),scrollHeight:primary.scrollHeight};})()"""
        )
        await screenshot("text-media-retrieval-detailed-collapsed-desktop.png")
        await evaluate(
            "document.querySelector('.text-media-visual-intent-policy').scrollIntoView({block:'center'})"
        )
        await asyncio.sleep(0.2)
        await screenshot("text-media-settings-module-gap-desktop.png")
        await evaluate(
            "document.querySelector('#text-media-edit-rrf-k').value='777';"
            "document.querySelector('#text-media-retrieval-reset').click()"
        )
        await wait_for("!document.querySelector('#confirm-modal').classList.contains('hidden')")
        retrieval_reset_confirmation = await evaluate(
            """({title:document.querySelector('#confirm-title').textContent,message:document.querySelector('#confirm-message').textContent,danger:document.querySelector('#confirm-modal .confirm-modal').classList.contains('danger')})"""
        )
        await screenshot("text-media-retrieval-reset-confirmation-desktop.png")
        await evaluate("document.querySelector('#confirm-ok').click()")
        await wait_for("document.querySelector('#confirm-modal').classList.contains('hidden')")
        retrieval_reset_result = await evaluate(
            """({rrf:Number(document.querySelector('#text-media-edit-rrf-k').value),candidate:Number(document.querySelector('#text-media-edit-candidate-limit').value),rerankCandidate:Number(document.querySelector('#text-media-edit-rerank-candidate-limit').value),gate:document.querySelector('#text-media-edit-intent-gate').checked,status:document.querySelector('#text-media-retrieval-status').textContent})"""
        )
        await evaluate(
            "document.querySelector('#text-media-edit-modal .modal-dismiss').click()"
        )
        await wait_for(
            "document.querySelector('#text-media-edit-modal').classList.contains('hidden')"
        )
        await evaluate(
            "document.querySelector('.library-card[data-id=beileite_test] .manage-text-media').click()"
        )
        await wait_for("document.querySelector('.text-media-workspace-page')")
        await evaluate(
            "document.querySelector('[data-route=\"database:text_media_v1:search\"]').click()"
        )
        await wait_for("document.querySelector('#text-media-search-form')")
        mode_layouts = await evaluate(
            """(()=>{
              const inspect=(mode)=>{
                document.querySelector(`[data-text-media-retrieval-mode="${mode}"]`).click();
                const items=[...document.querySelectorAll('#text-media-search-form .text-media-search-parameter:not(.hidden)')];
                return {cards:items.length,columns:new Set(items.map(item=>Math.round(item.getBoundingClientRect().left))).size,widths:items.map(item=>Math.round(item.getBoundingClientRect().width))};
              };
              const result={standard:inspect('standard'),textOnly:inspect('text_only'),mediaOnly:inspect('media_only')};
              document.querySelector('[data-text-media-retrieval-mode="standard"]').click();
              return result;
            })()"""
        )
        await evaluate(
            "document.querySelector('#text-media-search-query').value='描述一下你的立绘';"
            "document.querySelector('#text-media-search-form').requestSubmit()"
        )
        await wait_for(
            "document.querySelectorAll('#text-media-search-decisions .text-media-decision').length>0",
            timeout=30,
        )
        embedding_view = await evaluate(
            """({
              active:document.querySelector('#text-media-search-view-embedding').classList.contains('active'),
              rerankAvailable:!document.querySelector('#text-media-search-view-rerank').disabled,
              resultCount:document.querySelectorAll('#text-media-search-results .text-media-result').length,
              hasNoRerankFields:[...document.querySelectorAll('#text-media-search-results .text-media-result')].every(item=>!item.innerText.includes('Rerank'))
            })"""
        )
        await evaluate(
            "document.querySelector('#text-media-search-view-rerank').click()"
        )
        await wait_for(
            "document.querySelector('#text-media-search-view-rerank').classList.contains('active')"
        )
        await asyncio.sleep(0.5)
        desktop_search = await evaluate(
            """({
              publicInputs:['text-media-search-top-k','text-media-search-confidence-threshold','text-media-search-score-threshold','text-media-search-max-outputs'].filter(id=>document.getElementById(id)).length,
              parameterCards:document.querySelectorAll('#text-media-search-form .text-media-search-parameter').length,
              parameterColumns:new Set([...document.querySelectorAll('#text-media-search-form .text-media-search-parameter:not(.hidden)')].map(item=>Math.round(item.getBoundingClientRect().left))).size,
              parameterMaxWidth:Math.max(...[...document.querySelectorAll('#text-media-search-form .text-media-search-parameter:not(.hidden)')].map(item=>Math.round(item.getBoundingClientRect().width))),
              parameterMaxHeight:Math.max(...[...document.querySelectorAll('#text-media-search-form .text-media-search-parameter:not(.hidden)')].map(item=>Math.round(item.getBoundingClientRect().height))),
              parameterMaxInternalGap:Math.max(...[...document.querySelectorAll('#text-media-search-form .text-media-search-parameter:not(.hidden)')].map(item=>Math.round(item.querySelector('input').getBoundingClientRect().left-item.querySelector('span').getBoundingClientRect().right))),
              parameterMinGroupGap:(items=>Math.min(...items.slice(1).map((item,index)=>Math.round(item.getBoundingClientRect().left-items[index].getBoundingClientRect().right))))([...document.querySelectorAll('#text-media-search-form .text-media-search-parameter:not(.hidden)')]),
              parameterInputsBordered:[...document.querySelectorAll('#text-media-search-form .text-media-search-parameter>input')].every(input=>parseFloat(getComputedStyle(input).borderTopWidth)>=1),
              parameterLabelsInline:[...document.querySelectorAll('#text-media-search-form .text-media-search-parameter:not(.hidden)')].every(item=>{const label=item.querySelector('span').getBoundingClientRect(),input=item.querySelector('input').getBoundingClientRect();return Math.abs((label.top+label.height/2)-(input.top+input.height/2))<=2}),
              actionDisplay:getComputedStyle(document.querySelector('.text-media-search-action-row')).display,
              actionSharesRow:(()=>{const action=document.querySelector('.text-media-search-action-row').getBoundingClientRect();const first=document.querySelector('#text-media-search-form .text-media-search-parameter:not(.hidden)').getBoundingClientRect();return action.top < first.bottom - 2 && action.bottom > first.top + 2;})(),
              actionInToolbar:document.querySelector('.text-media-search-action-row')?.parentElement?.classList.contains('text-media-search-parameter-grid')||false,
              rerankRole:document.querySelector('#text-media-search-rerank').getAttribute('role'),
              rerankSwitchWidth:Math.round(document.querySelector('.text-media-search-rerank-toggle .switch').getBoundingClientRect().width),
              rerankToggleBorderWidth:parseFloat(getComputedStyle(document.querySelector('.text-media-search-rerank-toggle')).borderTopWidth),
              rerankToggleGap:Math.round(document.querySelector('.text-media-search-rerank-toggle .switch').getBoundingClientRect().left-document.querySelector('.text-media-search-rerank-toggle strong').getBoundingClientRect().right),
              submitWidth:Math.round(document.querySelector('.text-media-search-submit').getBoundingClientRect().width),
              outputs:document.querySelectorAll('#text-media-search-media .text-media-search-media-card').length,
              decisions:document.querySelectorAll('#text-media-search-decisions .text-media-decision').length,
              diagnosticTerms:document.querySelectorAll('#text-media-search-decisions .text-media-decision dt').length,
              summaryTerms:[...document.querySelectorAll('#text-media-search-decisions .text-media-decision-summary')].map(item=>item.querySelectorAll('dt').length),
              detailTerms:[...document.querySelectorAll('#text-media-search-decisions .text-media-diagnostic-details')].map(item=>item.querySelectorAll('dt').length),
              diagnosticDetails:document.querySelectorAll('#text-media-search-decisions .text-media-diagnostic-details').length,
              allDetailsCollapsed:[...document.querySelectorAll('#text-media-search-decisions details')].every(item=>!item.open),
              pivotLabel:document.querySelector('[data-i18n=mediaRelevancePivot]')?.textContent||'',
              hasSubMillConfidence:document.querySelector('#text-media-search-decisions').innerText.includes('<0.001'),
              decisionListColumns:getComputedStyle(document.querySelector('#text-media-search-decisions')).gridTemplateColumns.split(' ').length,
              decisionColumns:getComputedStyle(document.querySelector('#text-media-search-decisions .text-media-decision')).gridTemplateColumns.split(' ').length,
              diagnostics:document.querySelector('#text-media-search-decisions').innerText,
              rerankSwitchChecked:document.querySelector('#text-media-search-rerank').checked,
              rerankSwitchDisabled:document.querySelector('#text-media-search-rerank').disabled,
              rerankStatus:document.querySelector('#text-media-search-rerank-status').innerText,
              rerankTextResults:[...document.querySelectorAll('#text-media-search-results .text-media-result')].every(item=>item.innerText.includes('Rerank')),
              boundIsolationLabels:['媒体证据范围','仅当前媒体绑定分块','媒体证据排名范围','按媒体独立排名','绑定分块总数','媒体候选分块数','媒体门控','具体对象锚点'].every(label=>document.querySelector('#text-media-search-decisions').textContent.includes(label)),
              rerankDetailLabels:['Rerank 原始分','Rerank 融合分','Rerank 置信度变化','校准强度来源','Rerank 校准降级原因'].every(label=>document.querySelector('#text-media-search-decisions').textContent.includes(label)),
              pivotDetailLabels:['原始相关度','枢轴校准后相关度','枢轴校准后直接相关度','结构衰减因数','格式衰减因数','内容衰减因数','负向压力保留因数'].every(label=>document.querySelector('#text-media-search-decisions').textContent.includes(label)),
              decisionPanelCollapsed:document.querySelector('#text-media-search-decisions-section').classList.contains('text-media-search-decisions-collapsed'),
              decisionToggleVisible:!document.querySelector('#text-media-search-decisions-toggle').classList.contains('hidden'),
              decisionToggleExpanded:document.querySelector('#text-media-search-decisions-toggle').getAttribute('aria-expanded'),
              decisionVisibleCount:[...document.querySelectorAll('#text-media-search-decisions .text-media-decision')].filter(item=>item.getBoundingClientRect().top < document.querySelector('#text-media-search-decisions').getBoundingClientRect().bottom - 1).length,
              collapsibleChunks:document.querySelectorAll('#text-media-search-results .text-media-result-collapsible').length,
              chunkCollapseValid:[...document.querySelectorAll('#text-media-search-results .text-media-result')].every(item=>{
                const primary=item.querySelector('.text-media-result-primary');
                const toggle=item.querySelector('.text-media-result-toggle');
                const shouldCollapse=primary.scrollHeight > Math.round(window.innerHeight / 3) + 24;
                return shouldCollapse
                  ? item.classList.contains('text-media-result-collapsed') && !toggle.classList.contains('hidden') && primary.clientHeight === Math.round(window.innerHeight / 3)
                  : !item.classList.contains('text-media-result-collapsible') && toggle.classList.contains('hidden');
              }),
              overflow:document.querySelector('.text-media-workspace-page').scrollWidth > document.querySelector('.text-media-workspace-page').clientWidth + 1
            })"""
        )
        original_theme = await evaluate("document.documentElement.dataset.theme")
        if original_theme != "dark":
            await evaluate("document.querySelector('#theme-toggle').click()")
            await asyncio.sleep(0.2)
        await screenshot("text-media-search-form-desktop-dark.png")
        if original_theme != "dark":
            await evaluate("document.querySelector('#theme-toggle').click()")
            await asyncio.sleep(0.2)
        await screenshot("text-media-search-desktop.png")
        await evaluate(
            "document.querySelector('#text-media-search-results .text-media-result-collapsible')?.scrollIntoView({block:'center'})"
        )
        await asyncio.sleep(0.2)
        await screenshot("text-media-search-chunk-collapsed-desktop.png")
        await evaluate(
            "document.querySelector('#text-media-search-results .text-media-result-collapsible .text-media-result-toggle')?.click()"
        )
        await asyncio.sleep(0.5)
        desktop_chunk_expanded = await evaluate(
            """(()=>{const item=document.querySelector('#text-media-search-results .text-media-result-collapsible');const primary=item?.querySelector('.text-media-result-primary');return {present:Boolean(item),expanded:item?.classList.contains('text-media-result-expanded')||false,aria:item?.querySelector('.text-media-result-toggle')?.getAttribute('aria-expanded')||'',fullyVisible:Boolean(primary)&&primary.clientHeight===primary.scrollHeight};})()"""
        )
        await screenshot("text-media-search-chunk-expanded-desktop.png")
        await evaluate(
            "document.querySelector('#text-media-search-results .text-media-result-collapsible .text-media-result-toggle')?.click();"
            "document.querySelector('#text-media-search-decisions-section')?.scrollIntoView({block:'start'})"
        )
        await asyncio.sleep(0.5)
        await screenshot("text-media-search-decisions-collapsed-desktop.png")
        await evaluate(
            "document.querySelector('#text-media-search-decisions-toggle')?.click()"
        )
        await asyncio.sleep(0.5)
        desktop_decisions_expanded = await evaluate(
            """(()=>{const panel=document.querySelector('#text-media-search-decisions-section');const content=document.querySelector('#text-media-search-decisions');const toggle=document.querySelector('#text-media-search-decisions-toggle');return {expanded:panel.classList.contains('text-media-search-decisions-expanded'),aria:toggle.getAttribute('aria-expanded'),fullyVisible:content.clientHeight===content.scrollHeight};})()"""
        )
        await screenshot("text-media-search-decisions-expanded-desktop.png")
        await evaluate(
            "document.querySelector('#text-media-search-decisions-toggle')?.click()"
        )
        await asyncio.sleep(0.5)
        await evaluate(
            "document.querySelector('#text-media-search-decisions .text-media-diagnostic-details').open=true;"
            "document.querySelector('#text-media-search-decisions .text-media-diagnostic-details').scrollIntoView({block:'center'})"
        )
        await asyncio.sleep(0.2)
        desktop_expanded = await evaluate(
            """({
              open:document.querySelector('#text-media-search-decisions .text-media-diagnostic-details').open,
              visible:document.querySelector('#text-media-search-decisions .text-media-diagnostic-details dl').getClientRects().length>0,
              overflow:document.querySelector('.text-media-workspace-page').scrollWidth > document.querySelector('.text-media-workspace-page').clientWidth + 1
            })"""
        )
        await screenshot("text-media-search-details-desktop.png")
        await evaluate(
            "document.querySelector('#text-media-search-decisions .text-media-diagnostic-details').open=false"
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
        await asyncio.sleep(0.3)
        await evaluate(
            "document.querySelector('#text-media-search-form').scrollIntoView({block:'start'})"
        )
        await asyncio.sleep(0.2)
        await screenshot("text-media-search-form-mobile.png")
        await evaluate(
            "document.querySelector('#text-media-search-decisions').scrollIntoView({block:'start'})"
        )
        await asyncio.sleep(0.2)
        mobile_search = await evaluate(
            """({
              bodyOverflow:document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
              modalOverflow:document.querySelector('.text-media-workspace-page').scrollWidth > document.querySelector('.text-media-workspace-page').clientWidth + 1,
              formColumns:getComputedStyle(document.querySelector('#text-media-search-form')).gridTemplateColumns,
              parameterColumns:new Set([...document.querySelectorAll('#text-media-search-form .text-media-search-parameter:not(.hidden)')].map(item=>Math.round(item.getBoundingClientRect().left))).size,
              actionDisplay:getComputedStyle(document.querySelector('.text-media-search-action-row')).display,
              actionWraps:document.querySelector('.text-media-search-action-row').getBoundingClientRect().width >= document.querySelector('.text-media-search-parameter-grid').getBoundingClientRect().width - 2,
              rerankSwitchVisible:document.querySelector('.text-media-search-rerank-toggle .switch').getClientRects().length>0,
              decisionPanelCollapsed:document.querySelector('#text-media-search-decisions-section').classList.contains('text-media-search-decisions-collapsed'),
              decisionVisibleCount:[...document.querySelectorAll('#text-media-search-decisions .text-media-decision')].filter(item=>item.getBoundingClientRect().top < document.querySelector('#text-media-search-decisions').getBoundingClientRect().bottom - 1).length,
              chunkCollapseValid:[...document.querySelectorAll('#text-media-search-results .text-media-result')].every(item=>{
                const primary=item.querySelector('.text-media-result-primary');
                const toggle=item.querySelector('.text-media-result-toggle');
                const shouldCollapse=primary.scrollHeight > Math.round(window.innerHeight / 3) + 24;
                return shouldCollapse
                  ? item.classList.contains('text-media-result-collapsed') && !toggle.classList.contains('hidden') && primary.clientHeight === Math.round(window.innerHeight / 3)
                  : !item.classList.contains('text-media-result-collapsible') && toggle.classList.contains('hidden');
              })
            })"""
        )
        await screenshot("text-media-search-mobile.png")
        await evaluate(
            "document.querySelector('#text-media-search-decisions .text-media-diagnostic-details').open=true"
        )
        await asyncio.sleep(0.2)
        mobile_expanded = await evaluate(
            """({
              open:document.querySelector('#text-media-search-decisions .text-media-diagnostic-details').open,
              bodyOverflow:document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
              modalOverflow:document.querySelector('.text-media-workspace-page').scrollWidth > document.querySelector('.text-media-workspace-page').clientWidth + 1
            })"""
        )
        await screenshot("text-media-search-details-mobile.png")
        await evaluate(
            "document.querySelector('#text-media-search-decisions .text-media-diagnostic-details').open=false"
        )
        await evaluate(
            "document.querySelector('#text-media-search-top-k').value='3';"
            "document.querySelector('#text-media-search-confidence-threshold').value='0.2';"
            "document.querySelector('#text-media-search-score-threshold').value='0.8';"
            "document.querySelector('#text-media-search-max-outputs').value='1';"
            "document.querySelector('#text-media-page-refresh').click()"
        )
        await asyncio.sleep(0.3)
        refresh_behavior = await evaluate(
            """({
              query:document.querySelector('#text-media-search-query').value,
              topK:Number(document.querySelector('#text-media-search-top-k').value),
              confidence:Number(document.querySelector('#text-media-search-confidence-threshold').value),
              score:Number(document.querySelector('#text-media-search-score-threshold').value),
              maxOutputs:Number(document.querySelector('#text-media-search-max-outputs').value),
              rerankChecked:document.querySelector('#text-media-search-rerank').checked,
              rerankDisabled:document.querySelector('#text-media-search-rerank').disabled,
              rerankStatus:document.querySelector('#text-media-search-rerank-status').textContent,
              outputs:document.querySelector('#text-media-search-media').childElementCount,
              decisions:document.querySelector('#text-media-search-decisions').childElementCount,
              results:document.querySelector('#text-media-search-results').childElementCount
            })"""
        )
        await evaluate(
            "document.querySelector('.nav[data-page=libraries]').click()"
        )
        await wait_for("document.querySelector('#page-libraries.active')")
        await evaluate(
            "document.querySelector('.database-category-tab[data-database-category=knowledge]')?.click()"
        )
        await wait_for(
            "document.querySelector('.library-card[data-id=beileite_test] .edit-text-media')"
        )
        await evaluate(
            "document.querySelector('.library-card[data-id=beileite_test] .edit-text-media').click()"
        )
        await wait_for(
            "!document.querySelector('#text-media-edit-modal').classList.contains('hidden')"
        )
        await evaluate(
            "document.querySelector('.text-media-retrieval-settings').scrollIntoView({block:'start'})"
        )
        await asyncio.sleep(0.2)
        mobile_edit = await evaluate(
            """({
              bodyOverflow:document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
              modalOverflow:document.querySelector('#text-media-edit-modal .modal').scrollWidth > document.querySelector('#text-media-edit-modal .modal').clientWidth + 1,
              settingsColumns:getComputedStyle(document.querySelector('.text-media-retrieval-settings-grid')).gridTemplateColumns,
              compact:document.querySelector('#text-media-retrieval-settings').dataset.layout==='compact',
              collapsed:document.querySelector('#text-media-retrieval-settings').classList.contains('text-media-retrieval-collapsed'),
              collapsedHeight:document.querySelector('#text-media-retrieval-primary').clientHeight,
              expectedHeight:Math.round(window.innerHeight*0.75),
              moduleGap:Math.round(document.querySelector('.text-media-visual-intent-policy').getBoundingClientRect().top-document.querySelector('#text-media-retrieval-settings').getBoundingClientRect().bottom)
            })"""
        )
        await screenshot("text-media-edit-mobile.png")
        await evaluate(
            "document.querySelector('.text-media-retrieval-toggle').scrollIntoView({block:'center'})"
        )
        await asyncio.sleep(0.2)
        mobile_intent_switch = await evaluate(
            """(()=>{const control=document.querySelector('#text-media-edit-intent-gate');const shell=document.querySelector('.text-media-intent-gate-switch');return {role:control.getAttribute('role'),width:Math.round(shell.getBoundingClientRect().width),visible:shell.getClientRects().length>0,bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,modalOverflow:document.querySelector('#text-media-edit-modal .modal').scrollWidth>document.querySelector('#text-media-edit-modal .modal').clientWidth+1};})()"""
        )
        await screenshot("text-media-intent-switch-mobile.png")
        await evaluate("document.querySelector('[data-retrieval-layout=detailed]').click()")
        await asyncio.sleep(0.5)
        mobile_edit_detailed = await evaluate(
            """(()=>{const section=document.querySelector('#text-media-retrieval-settings');return {layout:section.dataset.layout,columns:getComputedStyle(section.querySelector('.text-media-retrieval-settings-grid')).gridTemplateColumns.split(' ').length,visibleHints:[...section.querySelectorAll('.text-media-retrieval-settings-grid small')].filter(item=>item.getClientRects().length).length,collapsed:section.classList.contains('text-media-retrieval-collapsed'),bodyOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1,modalOverflow:document.querySelector('#text-media-edit-modal .modal').scrollWidth>document.querySelector('#text-media-edit-modal .modal').clientWidth+1};})()"""
        )
        await screenshot("text-media-retrieval-detailed-collapsed-mobile.png")
        await evaluate(
            "document.querySelector('.text-media-visual-intent-policy').scrollIntoView({block:'start'});"
            "document.querySelector('#text-media-edit-modal .modal').scrollTop-=72"
        )
        await asyncio.sleep(0.2)
        await screenshot("text-media-settings-module-gap-mobile.png")
        await evaluate(
            "document.querySelector('#text-media-edit-modal .modal-dismiss').click();"
            "document.querySelector('.nav[data-page=libraries]').click();"
            "document.querySelector('.database-category-tab[data-database-category=knowledge]')?.click();"
            "document.querySelector('#library-create').click()"
        )
        await wait_for(
            "document.querySelector('#database-type-cards [data-type=text_media_v1]')"
        )
        await evaluate(
            "document.querySelector('#database-type-cards [data-type=text_media_v1]').click()"
        )
        await wait_for(
            "!document.querySelector('#text-media-create-modal').classList.contains('hidden')"
        )
        create_rerank = await evaluate(
            """({
              present:Boolean(document.querySelector('#text-media-create-rerank-provider')),
              value:document.querySelector('#text-media-create-rerank-provider').value,
              options:document.querySelectorAll('#text-media-create-rerank-provider option').length,
              overflow:document.querySelector('#text-media-create-modal .modal').scrollWidth > document.querySelector('#text-media-create-modal .modal').clientWidth + 1
            })"""
        )
        await screenshot("text-media-create-rerank-mobile.png")
    return {
        "library_card": library_card,
        "desktop_edit": desktop_edit,
        "desktop_retrieval_compact": desktop_retrieval_compact,
        "desktop_retrieval_expanded": desktop_retrieval_expanded,
        "desktop_retrieval_detailed": desktop_retrieval_detailed,
        "desktop_retrieval_detailed_collapsed": desktop_retrieval_detailed_collapsed,
        "retrieval_reset_confirmation": retrieval_reset_confirmation,
        "retrieval_reset_result": retrieval_reset_result,
        "mode_layouts": mode_layouts,
        "embedding_view": embedding_view,
        "desktop_search": desktop_search,
        "desktop_chunk_expanded": desktop_chunk_expanded,
        "desktop_decisions_expanded": desktop_decisions_expanded,
        "desktop_expanded": desktop_expanded,
        "refresh_behavior": refresh_behavior,
        "mobile_search": mobile_search,
        "mobile_expanded": mobile_expanded,
        "mobile_edit": mobile_edit,
        "mobile_intent_switch": mobile_intent_switch,
        "mobile_edit_detailed": mobile_edit_detailed,
        "create_rerank": create_rerank,
        "runtime_exceptions": len(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devtools-url", default="http://127.0.0.1:9223")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
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
        and result["library_card"]["rerank"]
        and result["desktop_edit"]["settings"] >= 24
        and result["desktop_edit"]["rerankProvider"] == "vllm_rerank"
        and result["desktop_edit"]["rerankCandidate"] == 10
        and result["desktop_edit"]["rerankFusionWeight"] == 0.30
        and result["desktop_edit"]["rerankRankBonus"] == 0
        and result["desktop_edit"]["rerankReliabilityExponent"] == 1.5
        and result["desktop_edit"]["textLexicalBoost"] == 0.6
        and result["desktop_edit"]["rerankFusionWeightValid"]
        and result["desktop_edit"]["candidate"] == 30
        and result["desktop_edit"]["floor"] == 0.35
        and result["desktop_edit"]["semanticWeight"] == 0.8
        and result["desktop_edit"]["lexicalBoost"] == 0.3
        and result["desktop_edit"]["thresholdFallback"] == 0.35
        and result["desktop_edit"]["thresholdEvidenceLimit"] == 5
        and result["desktop_edit"]["thresholdRankExponent"] == 1.5
        and result["desktop_edit"]["thresholdNegativeExponent"] == 1.5
        and result["desktop_edit"]["thresholdReinforcement"] == 0.7
        and result["desktop_edit"]["thresholdWeakening"] == 0.35
        and result["desktop_edit"]["pivotNegativeFloor"] == 0.05
        and result["desktop_edit"]["formatMismatchFactor"] == 0.1
        and result["desktop_edit"]["contentMismatchFactor"] == 0.1
        and result["desktop_edit"]["gate"]
        and result["desktop_edit"]["gateRole"] == "switch"
        and result["desktop_edit"]["gateSwitchWidth"] == 48
        and result["desktop_edit"]["moduleGap"] >= 16
        and not result["desktop_edit"]["overflow"]
        and result["desktop_retrieval_compact"]["layout"] == "compact"
        and result["desktop_retrieval_compact"]["columns"] == 3
        and result["desktop_retrieval_compact"]["visibleHints"] == 0
        and result["desktop_retrieval_compact"]["collapsed"]
        and result["desktop_retrieval_compact"]["toggleVisible"]
        and result["desktop_retrieval_compact"]["aria"] == "false"
        and result["desktop_retrieval_compact"]["clientHeight"]
        == result["desktop_retrieval_compact"]["expectedHeight"]
        and result["desktop_retrieval_compact"]["scrollHeight"]
        > result["desktop_retrieval_compact"]["clientHeight"]
        and result["desktop_retrieval_expanded"]
        == {"expanded": True, "aria": "true", "fullyVisible": True}
        and result["desktop_retrieval_detailed"]["layout"] == "detailed"
        and result["desktop_retrieval_detailed"]["columns"] == 1
        and result["desktop_retrieval_detailed"]["visibleHints"] == 36
        and result["desktop_retrieval_detailed"]["expanded"]
        and result["desktop_retrieval_detailed"]["fullyVisible"]
        and result["desktop_retrieval_detailed_collapsed"]["collapsed"]
        and result["desktop_retrieval_detailed_collapsed"]["clientHeight"]
        == result["desktop_retrieval_detailed_collapsed"]["expectedHeight"]
        and result["desktop_retrieval_detailed_collapsed"]["scrollHeight"]
        > result["desktop_retrieval_detailed_collapsed"]["clientHeight"]
        and result["retrieval_reset_confirmation"]["danger"]
        and "检索" in result["retrieval_reset_confirmation"]["title"]
        and "保存" in result["retrieval_reset_confirmation"]["message"]
        and result["retrieval_reset_result"]["rrf"] == 60
        and result["retrieval_reset_result"]["candidate"] == 30
        and result["retrieval_reset_result"]["rerankCandidate"] == 10
        and result["retrieval_reset_result"]["gate"]
        and "保存" in result["retrieval_reset_result"]["status"]
        and result["mode_layouts"]["standard"]["cards"] == 4
        and result["mode_layouts"]["standard"]["columns"] == 4
        and result["mode_layouts"]["textOnly"]["cards"] == 1
        and result["mode_layouts"]["textOnly"]["columns"] == 1
        and result["mode_layouts"]["textOnly"]["widths"][0] <= 240
        and result["mode_layouts"]["mediaOnly"]["cards"] == 3
        and result["mode_layouts"]["mediaOnly"]["columns"] == 3
        and max(result["mode_layouts"]["mediaOnly"]["widths"]) <= 240
        and result["desktop_search"]["publicInputs"] == 4
        and result["desktop_search"]["parameterCards"] == 4
        and result["desktop_search"]["parameterColumns"] == 4
        and result["desktop_search"]["parameterMaxWidth"] <= 240
        and result["desktop_search"]["parameterMaxHeight"] <= 40
        and result["desktop_search"]["parameterMaxInternalGap"] <= 8
        and result["desktop_search"]["parameterMinGroupGap"] >= 18
        and result["desktop_search"]["parameterInputsBordered"]
        and result["desktop_search"]["parameterLabelsInline"]
        and result["desktop_search"]["actionDisplay"] == "flex"
        and result["desktop_search"]["actionSharesRow"]
        and result["desktop_search"]["actionInToolbar"]
        and result["desktop_search"]["rerankRole"] == "switch"
        and result["desktop_search"]["rerankSwitchWidth"] == 48
        and result["desktop_search"]["rerankToggleBorderWidth"] == 0
        and result["desktop_search"]["rerankToggleGap"] <= 10
        and result["desktop_search"]["submitWidth"] >= 160
        and result["embedding_view"] == {
            "active": True,
            "rerankAvailable": True,
            "resultCount": 10,
            "hasNoRerankFields": True,
        }
        and result["desktop_search"]["outputs"] >= 1
        and result["desktop_search"]["decisions"] >= 4
        and result["desktop_search"]["diagnosticTerms"] >= 10
        and all(count == 8 for count in result["desktop_search"]["summaryTerms"])
        and all(count >= 16 for count in result["desktop_search"]["detailTerms"])
        and result["desktop_search"]["diagnosticDetails"]
        == result["desktop_search"]["decisions"]
        and result["desktop_search"]["allDetailsCollapsed"]
        and result["desktop_search"]["rerankSwitchChecked"]
        and not result["desktop_search"]["rerankSwitchDisabled"]
        and "Rerank" in result["desktop_search"]["rerankStatus"]
        and result["desktop_search"]["rerankTextResults"]
        and result["desktop_search"]["boundIsolationLabels"]
        and result["desktop_search"]["rerankDetailLabels"]
        and result["desktop_search"]["pivotDetailLabels"]
        and "媒体相关性枢轴" in result["desktop_search"]["pivotLabel"]
        and result["desktop_search"]["hasSubMillConfidence"]
        and result["desktop_search"]["decisionListColumns"] == 1
        and result["desktop_search"]["decisionColumns"] == 3
        and result["desktop_search"]["decisionPanelCollapsed"]
        and result["desktop_search"]["decisionToggleVisible"]
        and result["desktop_search"]["decisionToggleExpanded"] == "false"
        and result["desktop_search"]["decisionVisibleCount"] == 4
        and result["desktop_search"]["collapsibleChunks"] >= 1
        and result["desktop_search"]["chunkCollapseValid"]
        and result["desktop_chunk_expanded"] == {
            "present": True,
            "expanded": True,
            "aria": "true",
            "fullyVisible": True,
        }
        and result["desktop_decisions_expanded"] == {
            "expanded": True,
            "aria": "true",
            "fullyVisible": True,
        }
        and result["desktop_expanded"] == {
            "open": True,
            "visible": True,
            "overflow": False,
        }
        and result["refresh_behavior"] == {
            "query": "描述一下你的立绘",
            "topK": 10,
            "confidence": 0.6,
            "score": 0.35,
            "maxOutputs": 5,
            "rerankChecked": True,
            "rerankDisabled": False,
            "rerankStatus": "",
            "outputs": 0,
            "decisions": 0,
            "results": 0,
        }
        and not result["desktop_search"]["overflow"]
        and not result["mobile_search"]["bodyOverflow"]
        and not result["mobile_search"]["modalOverflow"]
        and result["mobile_search"]["parameterColumns"] == 1
        and result["mobile_search"]["actionDisplay"] == "flex"
        and result["mobile_search"]["actionWraps"]
        and result["mobile_search"]["rerankSwitchVisible"]
        and result["mobile_search"]["decisionPanelCollapsed"]
        and result["mobile_search"]["decisionVisibleCount"] == 4
        and result["mobile_search"]["chunkCollapseValid"]
        and result["mobile_expanded"] == {
            "open": True,
            "bodyOverflow": False,
            "modalOverflow": False,
        }
        and not result["mobile_edit"]["bodyOverflow"]
        and not result["mobile_edit"]["modalOverflow"]
        and result["mobile_edit"]["compact"]
        and result["mobile_edit"]["collapsed"]
        and result["mobile_edit"]["moduleGap"] >= 16
        and abs(
            result["mobile_edit"]["collapsedHeight"]
            - result["mobile_edit"]["expectedHeight"]
        ) <= 3
        and result["mobile_intent_switch"]
        == {
            "role": "switch",
            "width": 48,
            "visible": True,
            "bodyOverflow": False,
            "modalOverflow": False,
        }
        and result["mobile_edit_detailed"]["layout"] == "detailed"
        and result["mobile_edit_detailed"]["columns"] == 1
        and result["mobile_edit_detailed"]["visibleHints"] == 36
        and result["mobile_edit_detailed"]["collapsed"]
        and not result["mobile_edit_detailed"]["bodyOverflow"]
        and not result["mobile_edit_detailed"]["modalOverflow"]
        and result["create_rerank"]["present"]
        and result["create_rerank"]["value"] == ""
        and result["create_rerank"]["options"] >= 2
        and not result["create_rerank"]["overflow"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

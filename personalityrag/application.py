from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from . import __version__
from .application_context import ApplicationContext, activate_context, reset_context
from .http_middleware import (
    RequestContextMiddleware,
    adapter_busy_guard,
    static_asset_cache_policy,
)
from .logger import logger


def create_app(context: ApplicationContext | None = None) -> FastAPI:
    context = context or ApplicationContext.create()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        token = activate_context(context)
        try:
            logger.info(
                "服务启动：host=%s webui_port=%s access_port=%s api_key_fp=%s",
                context.config.host,
                os.environ.get("PERSONALITYRAG_ACTUAL_PORT")
                or context.config.port,
                os.environ.get("PERSONALITYRAG_ACCESS_ACTUAL_PORT")
                or context.config.access_port,
                context.config.api_key_fingerprint,
            )
            for warning_key in (
                "PERSONALITYRAG_WEBUI_PORT_FALLBACK_WARNING",
                "PERSONALITYRAG_ACCESS_PORT_FALLBACK_WARNING",
                "PERSONALITYRAG_PORT_FALLBACK_WARNING",
            ):
                if os.environ.get(warning_key):
                    logger.warning(os.environ[warning_key])
            await context.manager.initialize()
            logger.info("服务初始化完成：默认记忆库与模型提供商已加载")
            yield
        finally:
            logger.info("服务正在关闭：释放记忆库 runtime 与模型提供商连接")
            await context.manager.close()
            logger.info("服务已关闭")
            reset_context(token)

    app = FastAPI(
        title="PersonalityRAG",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.context = context
    app.mount("/static", StaticFiles(directory=context.static_dir), name="static")

    from .routes import ROUTERS

    for router in ROUTERS:
        app.include_router(router)
    app.middleware("http")(static_asset_cache_policy)
    app.middleware("http")(adapter_busy_guard)
    app.add_middleware(GZipMiddleware, minimum_size=500)
    app.add_middleware(RequestContextMiddleware, context=context)
    return app

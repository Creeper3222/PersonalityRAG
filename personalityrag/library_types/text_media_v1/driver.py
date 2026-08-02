from __future__ import annotations

import hmac
from typing import Iterable

from fastapi import APIRouter

from ...database_types import (
    DATABASE_CATEGORY_KNOWLEDGE,
    TEXT_MEDIA_V1_TYPE,
    DatabaseDriverContext,
    DatabaseTypeDescriptor,
    derive_domain_separated_access_key,
)


class TextMediaV1Driver:
    descriptor = DatabaseTypeDescriptor(
        id=TEXT_MEDIA_V1_TYPE,
        category=DATABASE_CATEGORY_KNOWLEDGE,
        display_name="文本媒体知识库 v1",
        description="以文本检索为核心，并把命中文本与规范化图片附件一同返回。",
        icon="/static/icons/text-media-v1.svg",
        capabilities=(
            "access_key",
            "adapter_access",
            "backup",
            "content_management",
            "copy",
            "image_assets",
            "index_rebuild",
            "search",
            "tmkb_export",
            "tmkb_import",
        ),
        key_prefix="pkb-",
        key_derivation_id="text-media-v1-domain-hmac-sha256-v1",
    )

    def resource_key(self, database_id: str) -> str:
        return f"{TEXT_MEDIA_V1_TYPE}:{database_id}"

    def create_manager(self, context: DatabaseDriverContext):
        from .manager import TextMediaV1Manager

        return TextMediaV1Manager(context)

    def api_routers(self) -> Iterable[APIRouter]:
        from .api import router
        from ...routes.libraries import knowledge_adapter_router

        return (router, knowledge_adapter_router)

    def derive_access_key(self, root_secret: str, database_id: str) -> str:
        return derive_domain_separated_access_key(
            root_secret=root_secret,
            database_type=TEXT_MEDIA_V1_TYPE,
            database_id=database_id,
            prefix="pkb-",
        )

    def verify_access_key(
        self, root_secret: str, database_id: str, candidate: str | None
    ) -> bool:
        if not candidate or not candidate.startswith("pkb-"):
            return False
        return hmac.compare_digest(
            candidate, self.derive_access_key(root_secret, database_id)
        )


driver = TextMediaV1Driver()
